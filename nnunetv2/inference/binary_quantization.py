import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict, OrderedDict
import math

from dataclasses import dataclass, asdict
import json
import os
from pathlib import Path

from dynamic_network_architectures.building_blocks.simple_conv_blocks import ConvDropoutNormReLU
from dynamic_network_architectures.building_blocks.unet_decoder import UNetDecoder
from dynamic_network_architectures.architectures.unet import PlainConvUNet
from dynamic_network_architectures.building_blocks.plain_conv_encoder import PlainConvEncoder
# ---------- 1) Fuse Conv + InstanceNorm (eval) ----------
def fuse_conv_instancenorm(conv: nn.Conv3d, inst: nn.InstanceNorm3d):
    assert isinstance(conv, nn.Conv3d)
    assert isinstance(inst, nn.InstanceNorm3d)

    # InstanceNorm must be in eval mode for running stats
    assert not inst.training, "InstanceNorm must be in eval() mode"

    # --- Extract InstanceNorm params ---
    rm = inst.running_mean            # [C]
    rv = inst.running_var             # [C]
    eps = inst.eps

    gamma = inst.weight if inst.weight is not None else torch.ones_like(rm)
    beta  = inst.bias   if inst.bias   is not None else torch.zeros_like(rm)

    sigma = torch.sqrt(rv + eps)      # [C]
    scale = gamma / sigma             # [C]
    bias_shift = beta - gamma * rm / sigma

    # --- Extract Conv params ---
    W = conv.weight.detach()          # [C_out, C_in, kD, kH, kW]
    if conv.bias is not None:
        b = conv.bias.detach()
    else:
        b = torch.zeros(W.shape[0], device=W.device)

    # --- Fuse ---
    W_fused = W * scale.view(-1, 1, 1, 1, 1)
    b_fused = b * scale + bias_shift

    # --- Create NEW Conv3d ---
    fused_conv = nn.Conv3d(
        in_channels=conv.in_channels,
        out_channels=conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=True,
        padding_mode=conv.padding_mode,
    )

    fused_conv.weight = nn.Parameter(W_fused)
    fused_conv.bias   = nn.Parameter(b_fused)

    return fused_conv

# ---------- 2) Collect activation ranges per-tensor (calibration) ----------
def collect_act_ranges(model, dataloader, device, n_batches=32):
    model.eval()
    ranges = defaultdict(lambda: [float('inf'), float('-inf')])  # min, max
    hooks = []

    def make_hook(name):
        def hook(module, inp, out):
            # out may be tensor or tuple
            if isinstance(out, torch.Tensor):
                t = out.detach().to('cpu')
                ranges[name][0] = min(ranges[name][0], float(t.min()))
                ranges[name][1] = max(ranges[name][1], float(t.max()))
            else:
                # iterate
                for j, x in enumerate(out):
                    t = x.detach().to('cpu')
                    key = f"{name}_{j}"
                    ranges[key][0] = min(ranges[key][0], float(t.min()))
                    ranges[key][1] = max(ranges[key][1], float(t.max()))
        return hook

    # Register hooks on layers that produce activations we need (Conv, ReLU, etc.)
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv3d, nn.ReLU, nn.LeakyReLU, nn.InstanceNorm3d, nn.MaxPool3d, nn.Upsample, nn.ConvTranspose3d)):
            hooks.append(module.register_forward_hook(make_hook(name)))

    with torch.no_grad():
        count = 0
        for batch in dataloader:
            x = batch['data'].to(device)
            model(x)
            count += 1
            if count >= n_batches:
                break

    for h in hooks: h.remove()
    return ranges  # dict: name -> [min, max]

# ---------- 3) Choose quantization scales ----------
def choose_activation_scale(min_val, max_val, symmetric=True):
    # Simple symmetric quantization (signed int8): qmin=-128, qmax=127
    if symmetric:
        a = max(abs(min_val), abs(max_val))
        qmin, qmax = -128, 127
        scale = a / qmax if a != 0 else 1e-8
        bias = 0.0
    else:
        qmin, qmax = 0, 255
        scale = (max_val - min_val) / (qmax - qmin) if max_val > min_val else 1e-8
        bias = min_val
    return float(scale), float(bias)

def choose_weight_scale_per_channel(W_fused):
    a = W_fused.abs().max()
    scale = float(a / 127.0) if a != 0 else 1e-8
    return scale  # tensor [C_out]

# ---------- 4) Quantize weights and biases ----------
def quantize_weights(W_fused, w_scale, ifm_scale, act_scale, is_conv3d = True, is_bias_by_ifm = False, ifm_bias = 0.0):
    # W_fused: [C_out, C_in, ...]
    # w_scale : int8
    # ifm_scale : int8 / uint8
    # act_scale : int8 / uint8
    # how to MAC & shift down to a reasonable amount?
    # ifm = ifm_scale * (int8 range) 

    # ifm / ifm_scale
    # wgt / (M * w_scale)
    # shift down by (s)
    # act / act_scale 

    # determine shift amount
    S = act_scale / (ifm_scale * w_scale)
    for s in range(32):
        if 1 <= S * (1 << s) and 2 > S * (1 << s):
            shift_amount = s 
            break
    M = S * (1<<(shift_amount))
    new_w_scale = w_scale * M 
    Wq = torch.empty_like(W_fused, dtype=torch.int8)
    Wq = torch.clamp((W_fused / new_w_scale).round(), -127, 127).to(torch.int8)

    if is_bias_by_ifm:
        if is_conv3d:
            weight_sum = W_fused.sum(dim = (1,2,3,4))
        else: 
            weight_sum = W_fused.sum(dim = (0,2,3,4))
        bias_by_ifm = ifm_bias * weight_sum 
    return Wq, new_w_scale, shift_amount, bias_by_ifm

def quantize_bias(b_fused, act_scale, shift_amount, is_bias_by_ifm = False, bias_by_ifm = None, is_include_act_bias = False, act_bias = 0.0):
    # b_fused: [C_out], act_scale: scalar (float), w_scales: [C_out]
    bias_int32 = torch.empty_like(b_fused, dtype=torch.int32)
    for c in range(b_fused.shape[0]):
        denom = act_scale
        bias_int32[c] = int(round((float(b_fused[c] + (bias_by_ifm[c] if is_bias_by_ifm else 0.0) - (act_bias if is_include_act_bias else 0.0)) * (1 << shift_amount))/ denom)) 
        # to approximate round by rightshift, add 0.5
        bias_int32[c] += (1 << (shift_amount - 1))
    return bias_int32


# ---------- 6) Integer conv emulation (per-output-channel weight scales, activation scale) ----------
def conv3dnormrelu(x_int, Wq, bias_int32, stride, shift_amount, is_leaky_relu):
    # x_int: int8 or int32 (HWC)... shape [B, C_in, D, H, W]
    # Wq: int8 weights [C_out, C_in, kD, kH, kW]
    # bias_int32: int32 per out channel in accumulation domain where accum_scale = act_in_scale * w_scales[c]
    # We'll compute int32 accum, then requantize to out_scale_target.
    # Compute multiplier & shift for each channel:
    C_out = Wq.shape[0]
    kernel_size = Wq.shape[2]

    # Do conv using int32 accum (naive; uses torch conv but with float -> int workaround)
    # We'll compute per-channel conv by converting Wq->int32 and x_int->int32 and using conv (float ops)
    # For emulator: do float conv with integer tensors cast to float then round after scaling.
    x_f = x_int.to(torch.float32)
    W_f = Wq.to(torch.float32)
    # naive conv (use F.conv3d)
    acc_f = F.conv3d(x_f, W_f, bias=None, stride=stride, padding = 1 if kernel_size >= 3 else 0)  # floating conv of integer-valued input
    # But acc_f currently equals sum(Wq * x_int) in float. cast to int32 (simulate exact int32)
    acc_int32 = torch.round(acc_f).to(torch.int32)
    # add bias
    acc_int32 = acc_int32 + bias_int32.view(1, -1, 1, 1, 1)
    # requantize per channel
    B, C, D, H, W = acc_int32.shape
    
    out_int = torch.empty_like(acc_int32, dtype=torch.int8)
    pos = acc_int32 >> shift_amount

    # Negative path: divide by 8 → extra right shift by 3
    # already added (1 << (shift_amount - 1)) -> append the rest
    neg = (acc_int32 + (7 << (shift_amount - 1))) >> (shift_amount + 3)

    if is_leaky_relu:
        out_int32 = torch.where(acc_int32 >= 0, pos, neg)
    else:
        out_int32 = pos
    
    # Saturate to int8 range
    out_int32 = torch.clamp(out_int32, -128, 127)

    out_int = out_int32.to(torch.int8)
    return out_int

# ---------- 6) Integer conv emulation (per-output-channel weight scales, activation scale) ----------
def convtranspose3dnormrelu(x_int, Wq, bias_int32, stride, shift_amount, is_leaky_relu):
    # x_int: int8 or int32 (HWC)... shape [B, C_in, D, H, W]
    # Wq: int8 weights [C_in,C_out, kD, kH, kW]
    # bias_int32: int32 per out channel in accumulation domain where accum_scale = act_in_scale * w_scales[c]
    # We'll compute int32 accum, then requantize to out_scale_target.
    # Compute multiplier & shift for each channel:
    C_out = Wq.shape[1]

    # Do conv using int32 accum (naive; uses torch conv but with float -> int workaround)
    # We'll compute per-channel conv by converting Wq->int32 and x_int->int32 and using conv (float ops)
    # For emulator: do float conv with integer tensors cast to float then round after scaling.
    x_f = x_int.to(torch.float32)
    W_f = Wq.to(torch.float32)
    # naive conv (use F.conv3d)
    acc_f = F.conv_transpose3d(x_f, W_f, bias=None, stride=stride, padding=0)  # floating conv of integer-valued input
    # But acc_f currently equals sum(Wq * x_int) in float. cast to int32 (simulate exact int32)
    acc_int32 = torch.round(acc_f).to(torch.int32)
    # add bias
    acc_int32 = acc_int32 + bias_int32.view(1, -1, 1, 1, 1)
    # requantize per channel
    B, C, D, H, W = acc_int32.shape
    
    out_int = torch.empty_like(acc_int32, dtype=torch.int8)
    pos = acc_int32 >> shift_amount

    # Negative path: divide by 8 → extra right shift by 3
    # already added (1 << (shift_amount - 1)) -> append the rest
    neg = (acc_int32 + (7 << (shift_amount - 1))) >> (shift_amount + 3)

    if is_leaky_relu:
        out_int32 = torch.where(acc_int32 >= 0, pos, neg)
    else:
        out_int32 = pos
    
    # Saturate to int8 range
    out_int32 = torch.clamp(out_int32, -128, 127)

    out_int = out_int32.to(torch.int8)
    return out_int


# decoder : conv transpose => torch.cat => stages => ... => seg_layer (at the end : 4x32x1x1x1 weight)
def get_act_scale(activation_min_max_dict, act_idx, symmetric = True):
    min_val = activation_min_max_dict[act_idx]["min"]
    max_val = activation_min_max_dict[act_idx]["max"]
    act_scale, act_bias = choose_activation_scale(min_val, max_val, symmetric=symmetric)
    return act_scale, act_bias

def summarize_act_scales(encoder, decoder, act_scale_list, act_bias_list, activation_min_max_dict):
    # Stage 1. For each CONV, list the associated ifm_scale and act_scale
    route_dict = {}
    route_target_list = []
    act_cnt = 0
    for stage_idx, encoder_stage in enumerate(encoder.stages):
        if stage_idx >= 1:
            route_target_list.append(act_cnt - 1)
        for blocks_idx, stacked_conv_blocks in enumerate(encoder_stage):
            for idx, convdropoutnormrelu in enumerate(stacked_conv_blocks.convs):
                act_scale, act_bias = get_act_scale(activation_min_max_dict, act_cnt)
                act_scale_list.append(act_scale)
                act_bias_list.append(act_bias)
                act_cnt+=1
    
    for s in range(len(decoder.stages)):
        # append conv transpose layer output of decoder
        act_scale, act_bias = get_act_scale(activation_min_max_dict, act_cnt)
        # compare activation scale with skip connection
        concat_prev_act_scale = abs(act_scale_list[route_target_list[-(s+1)]])
        max_act_scale =  max(abs(act_scale), concat_prev_act_scale)
        act_scale_list[route_target_list[-(s+1)]] = max_act_scale
        act_scale_list.append(max_act_scale)
        act_bias_list.append(act_bias)
        route_dict[act_cnt + 1] = route_target_list[-(s+1)]
        act_cnt += 1

        
        # append stages of decoder
        for idx, conv in enumerate(decoder.stages[s].convs):
            act_scale, act_bias = get_act_scale(activation_min_max_dict, act_cnt)
            act_scale_list.append(act_scale)
            act_bias_list.append(act_bias)
            act_cnt += 1
        
        # append seg_layer of decoder
        if s == len(decoder.stages) - 1:
            act_scale, act_bias = get_act_scale(activation_min_max_dict, act_cnt, symmetric=False)
            act_scale_list.append(act_scale)
            act_bias_list.append(act_bias)
            act_cnt += 1
    return route_dict

@dataclass
class ConvOp:
    idx:int
    weight:object
    bias:object
    shift_amount:int
    is_leaky_relu:bool
    ifm_is_signed:bool
    is_conv3d:bool
    is_concat:bool
    concat_idx:int
    kernel_size:int
    stride:int

def save_convop(convop: ConvOp, path: str):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    # Save tensors
    torch.save(convop.weight, path / "weight.pt")
    torch.save(convop.bias, path / "bias.pt")

    # Save metadata (everything except tensors)
    meta = asdict(convop)
    meta.pop("weight")
    meta.pop("bias")

    with open(path / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

def load_convop(path: str, device="cpu") -> ConvOp:
    path = Path(path)

    weight = torch.load(path / "weight.pt", map_location=device)
    bias = torch.load(path / "bias.pt", map_location=device)

    with open(path / "meta.json") as f:
        meta = json.load(f)

    return ConvOp(
        weight=weight,
        bias=bias,
        **meta
    )

def analyze_convdropoutnormrelu(convdropoutnormrelu, act_scale_list, layer_idx, initial_ifm_scale, route_dict, is_first_layer=False, is_seg_layer = False, zero = 0.0):
    fused_conv = fuse_conv_instancenorm(convdropoutnormrelu.conv, convdropoutnormrelu.norm)

    W_fused = fused_conv.weight.data
    w_scale = choose_weight_scale_per_channel(W_fused)
    ifm_scale = initial_ifm_scale if layer_idx == 0 else act_scale_list[layer_idx - 1]
    Wq, new_w_scale, shift_amount = quantize_weights(W_fused, w_scale, ifm_scale, act_scale_list[layer_idx], True, is_first_layer, zero)
    bias_int32 = quantize_bias(fused_conv.bias.data, act_scale_list[layer_idx], shift_amount, is_first_layer, zero, is_seg_layer, zero)
    
    return ConvOp(
            idx = layer_idx, weight=W_fused, bias = bias_int32,
            shift_amount= shift_amount,
            is_leaky_relu=True, ifm_is_signed = False if layer_idx == 0 else True,
            is_conv3d = isinstance(convdropoutnormrelu.conv, nn.Conv3d),
            is_concat = True if layer_idx in route_dict else False,
            concat_idx = route_dict[layer_idx] if layer_idx in route_dict else 0,
            kernel_size = Wq.shape[2],
            stride = convdropoutnormrelu.conv.stride
        )

def analyze_seg(seg, act_scale_list, layer_idx, route_dict, is_first_layer=False, is_seg_layer = True, zero = 0.0):

    W_fused = seg.weight.data.clone()
    w_scale = choose_weight_scale_per_channel(W_fused)
    ifm_scale = act_scale_list[layer_idx - 1]
    Wq, new_w_scale, shift_amount = quantize_weights(W_fused, w_scale, ifm_scale, act_scale_list[layer_idx], True, is_first_layer, zero)
    bias_int32 = quantize_bias(seg.bias.data, act_scale_list[layer_idx], shift_amount, is_first_layer,zero, is_seg_layer, zero)
    
    return ConvOp(
            idx = layer_idx, weight=W_fused, bias = bias_int32,
            shift_amount= shift_amount,
            is_leaky_relu=False, ifm_is_signed = True,
            is_conv3d = isinstance(seg, nn.Conv3d),
            is_concat = True if layer_idx in route_dict else False,
            concat_idx = route_dict[layer_idx] if layer_idx in route_dict else 0,
            kernel_size = Wq.shape[2],
            stride = seg.stride
        )
                
def analyze_convtranspose(conv_transpose, act_scale_list, layer_idx, is_first_layer=False, is_seg_layer = False, zero = 0.0):
    W_fused = conv_transpose.weight.data.clone()
    w_scale = choose_weight_scale_per_channel(W_fused)
    ifm_scale = act_scale_list[layer_idx - 1]
    Wq, new_w_scale, shift_amount = quantize_weights(W_fused, w_scale, ifm_scale, act_scale_list[layer_idx], False, is_first_layer, zero)
    bias_int32 = quantize_bias(conv_transpose.bias.data, act_scale_list[layer_idx], shift_amount, is_first_layer, zero, is_seg_layer, zero)
    
    return ConvOp(
            idx = layer_idx, weight=W_fused, bias = bias_int32,
            shift_amount= shift_amount,
            is_leaky_relu=False, ifm_is_signed = True,
            is_conv3d = False,
            is_concat = False,
            concat_idx = 0,
            kernel_size = Wq.shape[2],
            stride = conv_transpose.stride
        )

def flatten_network(network, ifm_min, ifm_max, activation_min_max_dict):
    assert isinstance(network, PlainConvUNet)

    encoder = network.encoder
    decoder = network.decoder   
    assert isinstance(encoder, PlainConvEncoder)
    assert isinstance(decoder, UNetDecoder)

    ops = []

    initial_ifm_scale, ifm_bias = choose_activation_scale(ifm_min, ifm_max, symmetric=False)
    print("initial IFM scale = ", initial_ifm_scale)
    act_scale_list = []
    act_bias_list = []
    route_dict = summarize_act_scales(encoder, decoder, act_scale_list, act_bias_list, activation_min_max_dict)
    layer_idx = 0
    for stage_idx, encoder_stage in enumerate(encoder.stages):
        for blocks_idx, stacked_conv_blocks in enumerate(encoder_stage):
            for idx, convdropoutnormrelu in enumerate(stacked_conv_blocks.convs):
                ops.append(
                    analyze_convdropoutnormrelu(convdropoutnormrelu, act_scale_list, layer_idx, initial_ifm_scale, route_dict, True if layer_idx == 0 else False, False, ifm_bias)
                )
                layer_idx += 1
    for s in range(len(decoder.stages)):
        conv_transpose = decoder.transpconvs[s]
        ops.append(
            analyze_convtranspose(conv_transpose, act_scale_list, layer_idx)
        )
        layer_idx += 1

        for idx, conv in enumerate(decoder.stages[s].convs):
            ops.append(
                analyze_convdropoutnormrelu(conv, act_scale_list, layer_idx, initial_ifm_scale, route_dict)
            )
            layer_idx += 1
    
        if s == len(decoder.stages) - 1:
            seg_head = decoder.seg_layers[-1]
            ops.append(
                analyze_seg(seg_head, act_scale_list, layer_idx, route_dict, False, True, act_bias_list[layer_idx])
            )
            layer_idx+=1

    return ops, initial_ifm_scale, ifm_bias, act_scale_list[-1], act_bias_list[-1], route_dict

def save_model(ops, ifm_scale, ifm_bias, seg_scale, seg_bias, model_path = "./ckpt"):
    os.makedirs(model_path, exist_ok=True)
    for i, op in enumerate(ops):
        save_convop(op, f"{model_path}/convop_{i}")
    meta = {
        "ifm_scale": ifm_scale,
        "ifm_bias": ifm_bias,
        "seg_scale": seg_scale,
        "seg_bias": seg_bias,
        "num_ops": len(ops),
    }

    torch.save(meta, os.path.join(model_path, "meta.pt"))

def load_model(model_path="./ckpt", device="cpu"):
    # load metadata
    meta = torch.load(os.path.join(model_path, "meta.pt"), map_location=device)

    ifm_scale = meta["ifm_scale"]
    ifm_bias  = meta["ifm_bias"]
    seg_scale = meta["seg_scale"]
    seg_bias  = meta["seg_bias"]

    num_ops = meta["num_ops"]

    # load conv ops
    ops = []
    for i in range(num_ops):
        op = load_convop(
            os.path.join(model_path, f"convop_{i}.pt"),
            device=device
        )
        ops.append(op)

    return ops, ifm_scale, ifm_bias, seg_scale, seg_bias

def forward(ifm, ops, ifm_scale, ifm_bias, seg_scale, seg_bias):
    x = torch.round((ifm - ifm_bias) / ifm_scale).clamp(0, 255).to(torch.uint8)
    tensors = {}
    for op in ops:
        if op.is_concat:
            x = torch.cat((x, tensors[op.concat_idx]), 1)
        if op.is_conv3d:
            x = conv3dnormrelu(x, op.weight, op.bias, op.stride, op.shift_amount, op.is_leaky_relu)
        else:
            x = convtranspose3dnormrelu(x, op.weight, op.bias, op.stride, op.shift_amount, op.is_leaky_relu)
        tensors[op.idx] = x
    return x.float() * seg_scale + seg_bias







def compute_dice(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6):
    """
    pred, target: integer tensors with same shape
    """
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)

    intersection = torch.sum(pred * target)
    dice = (2. * intersection + eps) / (
        torch.sum(pred) + torch.sum(target) + eps
    )
    return dice.item()

def print_quant_summary(metrics):
    l1s = [m["l1"] for m in metrics]
    l2s = [m["l2"] for m in metrics]
    dices = [m["dice"] for m in metrics if "dice" in m]

    print("\n====== Quantization Evaluation Summary ======")
    print(f"Samples           : {len(metrics)}")
    print(f"Mean L1 error     : {sum(l1s) / len(l1s):.6e}")
    print(f"Mean L2 error     : {sum(l2s) / len(l2s):.6e}")

    if dices:
        print(f"Mean Dice score   : {sum(dices) / len(dices):.4f}")
        print(f"Min Dice score    : {min(dices):.4f}")
    print("============================================\n")