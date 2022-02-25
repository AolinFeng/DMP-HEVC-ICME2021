# DMP-HEVC-ICME2021
Depth Map Prediction for Fast Block Partitioning in HEVC

The project page for the paper:

Aolin Feng, Changsheng Gao, Li Li, Dong Liu, and Feng Wu."CNN-based Depth Map Prediction for Fast Block Partitioning in HEVC Intra Coding", IEEE International Conference on Multimedia and Expo (ICME), 2021. [OpenAccess](https://ieeexplore.ieee.org/abstract/document/9428069)

## Bietex
    @inproceedings{feng2021cnn,
      title={Cnn-Based Depth Map Prediction for Fast Block Partitioning in HEVC Intra Coding},
      author={Feng, Aolin and Gao, Changsheng and Li, Li and Liu, Dong and Wu, Feng},
      booktitle={2021 IEEE International Conference on Multimedia and Expo (ICME)},
      pages={1--6},
      year={2021},
      organization={IEEE}
    }

## Running Instructions

* Dependency: Pytroch, GPU (the code only can be run in GPU mode)
* Command: python dp_total_test.py > test.log

## Folder Instructions
* cfg: Encoding configuration file
* cfg/per-sequence: Sequence information file
* Test_Sequence_List.txt: Names of sequences to be tested, which correspond to the names of sequence information file
* codec: Encoder.exe + Decoder.exe
* models: Trained models for depth map prediction
* DepthFlag: Output split flags used for encoding acceleration
* output: Output bit stream 
* log: Output encoding and decoding logs

## Attention:
* The working directory mus be the current directory, and the folder names cannot be modified.
* To test a sequence:
   * Write the sequence information file strictly according to the format of existing files. Use '/' rather than '\\' in the file path.
   * Write all the names of test sequences in "Test_Sequence_List.txt".
* In the output log file of the main python script, the last line stores the network inference time, which shpuld be added to the total encoding time. 