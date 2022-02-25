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

1. Dependency: Pytroch, GPU (the code only can be run in GPU mode)
2. Command: python dp_total_test.py > test.log

## Folder Instructions
1. cfg: Encoding configuration file
2. cfg/per-sequence: Sequence information file
3. Test_Sequence_List.txt: Names of sequences to be tested, which correspond to the names of sequence information file
4. codec: Encoder.exe + Decoder.exe
5. models: Trained models for depth map prediction
6. DepthFlag: Generated split flags used for encoding acceleration
7. output: Bit stream 
8. log: Encoding and decoding logs

## Attention:
1. The working directory mus be the current directory, and the folder names cannot be modified.
2. To test a sequence:

   2.1 Write the sequence information file strictly according to the format of existing files. Use '/' rather than '\\' in the file path.
  
   2.2 Write all the names of test sequences in "Test_Sequence_List.txt".
3. In the output log file of the main code, the last line stores the network inference time, which shpuld be added to the total encoding time. 
