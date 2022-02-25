# *********************************
# complete dp algorithm test procedure
# FengAl
# *********************************
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f
# from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Dataset, TensorDataset
import time
# import PreModel.dp_model3 as dp_model
import dp_model

pool = nn.MaxPool2d(kernel_size=2)
raster2zscan4 = np.array([0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15], dtype=np.int8)

def remove_prefix(state_dict, prefix):
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_pretrain_model(current_model, pretrain_model):
    source_dict = torch.load(pretrain_model)
    if "state_dict" in source_dict.keys():
        source_dict = remove_prefix(source_dict['state_dict'], 'module.')
    else:
        source_dict = remove_prefix(source_dict, 'module.')
    dest_dict = current_model.state_dict()
    trained_dict = {k: v for k, v in source_dict.items() if
                    k in dest_dict and source_dict[k].shape == dest_dict[k].shape}
    dest_dict.update(trained_dict)
    current_model.load_state_dict(dest_dict)
    return current_model

def import_yuv420(file_path, width, height, frm_num, sub_sample_ratio=1, is10bit=False):
    fp = open(file_path, 'rb')
    pix_num = width * height
    sub_frm_num = (frm_num + sub_sample_ratio - 1) // sub_sample_ratio # actual frame number after downsampling
    if is10bit:
        data_type = np.uint16
    else:
        data_type = np.uint8
    y_temp = np.zeros(pix_num*sub_frm_num, dtype=data_type)
    u_temp = np.zeros(pix_num*sub_frm_num // 4, dtype=data_type)
    v_temp = np.zeros(pix_num*sub_frm_num // 4, dtype=data_type)
    for i in range(0, frm_num, sub_sample_ratio):
        if is10bit:
            fp.seek(i * pix_num * 3, 0)
        else:
            fp.seek(i * pix_num * 3 // 2, 0)
        subi = i // sub_sample_ratio
        y_temp[subi*pix_num : (subi+1)*pix_num] = np.fromfile(fp, dtype=data_type, count=pix_num, sep='')
        u_temp[subi*pix_num//4 : (subi+1)*pix_num//4] = np.fromfile(fp, dtype=data_type, count=pix_num//4, sep='')
        v_temp[subi*pix_num//4 : (subi+1)*pix_num//4] = np.fromfile(fp, dtype=data_type, count=pix_num//4, sep='')
    fp.close()
    y = y_temp.reshape((sub_frm_num, height, width))
    u = u_temp.reshape((sub_frm_num, height//2, width//2))
    v = v_temp.reshape((sub_frm_num, height//2, width//2))
    if is10bit:
        y = np.clip((y + 2) / 4, 0, 255).astype(np.uint8)
        u = np.clip((u + 2) / 4, 0, 255).astype(np.uint8)
        v = np.clip((v + 2) / 4, 0, 255).astype(np.uint8)
    return y, u, v  # return frm_num * H * W

def out_block_y(file_path, width, height, block_size, frm_num, sub_sample_ratio, is10bit):
    # return num_block * block_size * block_size
    y, _, _ = import_yuv420(file_path, width, height, frm_num, sub_sample_ratio, is10bit)
    sub_frm_num = y.shape[0]

    width_round = int(width // block_size * block_size)
    height_round = int(height // block_size * block_size)
    y = y[:, 0:height_round, 0:width_round]
    block_num_in_width = int(width_round / block_size)
    block_num_in_height = int(height_round / block_size)
    # print(block_num_in_width, block_num_in_height)
    #total_block_num = numfrm * Block_Num_in_Height * Block_Num_in_Width
    block_list = []
    for f_num in range(sub_frm_num):
        for i in range(block_num_in_height):
            for j in range(block_num_in_width):
                block_list.append(y[f_num, i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size])
    block_y = np.array(block_list)
    # print('shape of block_y', block_y.shape)
    return block_y

def load_ifo_from_cfg(cfg_path):
    fp = open(cfg_path)
    input_path = None
    bit_depth = None
    width = None
    height = None
    for line in fp:
        if "InputFile" in line:
            line = line.rstrip('\n').replace(' ', '').split('#')[0]
            loc = line.find(':')
            input_path = line[loc+1:]
        elif "InputBitDepth" in line:
            bit_depth = int(line.rstrip('\n').replace(' ', '').split('#')[0].split(':')[-1])
        elif "SourceWidth" in line:
            width = int(line.rstrip('\n').replace(' ', '').split('#')[0].split(':')[-1])
        elif "SourceHeight" in line:
            height = int(line.rstrip('\n').replace(' ', '').split('#')[0].split(':')[-1])
    if (input_path is None) or (bit_depth is None) or (width is None) or (height is None):
        print("Format of CFG error !!!!!!!!")
        return
    return input_path, bit_depth, width, height

def Load_DP_Seq(file_path, width, height, block_size, frm_num, sub_sample_ratio, is10bit):
    block_y = out_block_y(file_path, width, height, block_size, frm_num, sub_sample_ratio, is10bit)
    # print("Start loading dataset...")
    input_batch = np.expand_dims(block_y, 1)
    input_batch = torch.FloatTensor(input_batch)
    # print("Shape of input batch:", input_batch.shape)
    # print("Creating data loader...")
    dataset = TensorDataset(input_batch)
    dataLoader = DataLoader(dataset=dataset, num_workers=2, batch_size=args.batchSize, pin_memory=True, shuffle=False)
    return dataLoader, input_batch.shape[0]

def depth2flag(depth_map, depth):
    cu_size = 64 >> depth  # 64 32 16
    cu_map_size = 8 >> depth  # 8 4 2
    offset = 1 << depth  # 1 2 4
    block_num = depth_map.shape[0]
    # print(block_num)
    flag_list = []
    for k in range(block_num):
        temp = []
        for i in range(0, 8, cu_map_size):
            for j in range(0, 8, cu_map_size):
                if depth_map[k, i, j] > depth:
                    temp.append(1)
                else:
                    temp.append(0)
        for i in range(len(temp)):
            if depth == 2:
                flag_list.append(temp[raster2zscan4[i]])
            else:
                flag_list.append(temp[i])
    return np.array(flag_list, dtype=np.int8)


def get_flag(depth_map, save_name, qp):
    flag_map64 = depth2flag(depth_map, 0)
    flag_map32 = depth2flag(depth_map, 1)
    flag_map16 = depth2flag(depth_map, 2)
    save_name = os.path.join('./DepthFlag', save_name + '_Q' + str(qp))
    # print('save name:', save_name)
    out64_file = open(save_name + '_64.txt', 'w')
    out32_file = open(save_name + '_32.txt', 'w')
    out16_file = open(save_name + '_16.txt', 'w')
    for i in range(flag_map64.size):
        out64_file.write(str(flag_map64[i]) + '\n')
    for i in range(flag_map32.size):
        out32_file.write(str(flag_map32[i]) + '\n')
    for i in range(flag_map16.size):
        out16_file.write(str(flag_map16[i]) + '\n')
    out64_file.close()
    out32_file.close()
    out16_file.close()



@torch.no_grad()
def inference_HEVC(qp):
    Net = dp_model.DP_Net(1)
    net_path = './models/model_qp' + str(qp) + '.pkl'
    # Net.load_state_dict(torch.load(net_path))
    Net = load_pretrain_model(Net, net_path)
    Net = nn.DataParallel(Net).cuda()

    seqs_info_path = "Test_Sequence_List.txt"
    seqs_info_fp = open(seqs_info_path, 'r')
    seqs = []
    for line in seqs_info_fp:
        if line is None:
            break
        seqs.append(line.rstrip('\n'))
    for seq_name in seqs:
        start_time = time.time()
        cfg_path = os.path.join("./cfg/per-sequence", seq_name + ".cfg")
        print(seq_name)
        input_path, bit_depth, width, height = load_ifo_from_cfg(cfg_path)
        if bit_depth != 10:
            is10bit = False
        else:
            is10bit = True
        data_loader, input_size0 = Load_DP_Seq(input_path, width, height, block_size=64, frm_num=1, sub_sample_ratio=8, is10bit=is10bit)
        out_total_batch = torch.zeros(input_size0, 1, 8, 8)  # save network output
        batch_num = 0
        with torch.no_grad():
            for step, data in enumerate(data_loader):
                input_batch = data[0]
                # print('##########################################step:', step)
                # print('input batch size:', input_batch.size())
                # print('label batch size:', label_batch.size())
                input_batch = input_batch.cuda()
                # label_batch = label_batch.cuda()
                out_batch = Net(input_batch)
                out_total_batch[batch_num:batch_num + out_batch.size()[0], :, :, :] = out_batch.cpu()
                batch_num += out_batch.size()[0]
                del input_batch, out_batch
            print('out batch shape:', out_total_batch.shape)
            get_flag(torch.round(out_total_batch).numpy().squeeze(axis=1), input_path.rstrip('.yuv').split('/')[-1], qp)
        infe_time = time.time() - start_time
        enc_dec_test(seq_name, qp)
    return infe_time

def enc_dec_test(seq_name, qp):
    cur_path = os.getcwd()
    # print(os.path.join(cur_path, "codec"))
    exe_path = os.path.join(cur_path, "codec")
    cfg_path = os.path.join(cur_path, "cfg")
    log_path = os.path.join(cur_path, "log")
    out_path = os.path.join(cur_path, "output")
   
    for profile in ["dpfast", "anchor"]:
        bin_name = seq_name + "_intra_main_" + profile + "_Q" + str(qp) + ".bin"
        enc_log_name = "enc_" + seq_name + "_intra_main_" + profile + "_Q" + str(qp) + ".log"
        dec_log_name = "dec_" + seq_name + "_intra_main_" + profile + "_Q" + str(qp) + ".log"

        enc_order = exe_path + "/TAppEncoder_" + profile + ".exe -c " + cfg_path + "/encoder_intra_main.cfg -c " \
                + cfg_path + "/per-sequence/" + seq_name + ".cfg -q " + str(qp) + " -b " + out_path + "/" + bin_name + \
            " -ip 1 -f 1 -o \"\" --SEIDecodedPictureHash=1 > " + log_path + "/" + enc_log_name

        dec_order = exe_path + "/TAppDecoder_" + profile + ".exe -b " + out_path + "/" + bin_name + \
            " > " + log_path + "/" + dec_log_name
        
        # print(enc_order)
        # print(dec_order)
        os.system(enc_order)
        os.system(dec_order)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSize', default=200, type=int, help='batch size')
    # parser.add_argument('--qp', default=22, type=int, help='QP')
    args = parser.parse_args()
    
    for dir in ["DepthFlag", "log", "output"]:
        if not os.path.exists(dir):
            os.makedirs(dir)

    qp_list = [22, 27, 32, 37]
    total_infe_time = 0
    for qp in qp_list:
        infe_time = inference_HEVC(qp)
        total_infe_time += infe_time

    print("inference time: ", total_infe_time)





