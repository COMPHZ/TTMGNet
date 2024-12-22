# TTMGNet: TTMGNet: Tree Topology Mamba-Guided Network Collaborative Hierarchical Incremental Aggregation for Change Detection
⭐ This code has been completely released ⭐ 

⭐ our [article](https://doi.org/10.3390/rs16214068) ⭐ 

If our code is helpful to you, please cite:

```
@article{wang2024ttmgnet,
  title={TTMGNet: Tree Topology Mamba-Guided Network Collaborative Hierarchical Incremental Aggregation for Change Detection},
  author={Wang, Hongzhu and Ye, Zhaoyi and Xu, Chuan and Mei, Liye and Lei, Cheng and Wang, Du},
  journal={Remote Sensing},
  volume={16},
  number={21},
  pages={4068},
  year={2024},
  publisher={MDPI}
}
```
## To Train 
 ```
python train.py 
```

## To Test
Pre-trained model to be uploaded later

## Dataset
Downloading the LEVIR-CD dataset from [LEVIR-CD](https://pan.baidu.com/s/1L7EJCGMivXm4OayRzjqa8w?pwd=7jp3)
Downloading the WHU-CD dataset from [LEVIR-CD](https://pan.baidu.com/s/1e0WPuyQVZBIQTzGbuk64ag?pwd=7jp3)
Downloading the CL-CD dataset from [LEVIR-CD](https://pan.baidu.com/s/1rG2PXDvd95D8VmhPfaBi8g?pwd=7jp3)

## overall network

<p align="center"> <img src="Main Stream.png" width="90%"> </p>

## Results

### LEVIR Datasset
#### Qualitative result
<p align="center"> <img src="Fig/tno.png" width="90%"> </p>
- Four representative images of the TNO test set.In alphabetical order they are infrared image, visible image, GTF, FusionGAN, SDNet, RFN–Nest, U2Fusion, LRRNet, SwinFusion, CDDFuse, DATFuse, and GTMFuse.

#### Quantitative Results

|   **Methods**    |   **Precision**   |   **Recall**   | **F1** | **OA** | **mIOU** | **Kappa** |
|:----------------:|:---------:|:---------:|:---------:|:------------:|:-----------------------:|:---------:|
| **FC-EF**         |   0.7991     | 0.8284    |  0.8135     |   0.9580      |  0.8197     |  0.7899   |
|  **BITNet**  |   0.8732      |   0.9141  |  0.    |   0.      | 0.   |  0.8796    |
| **HFANet**       |   0.8336      | 0.9148    |   0.     |  0.       | 0.    | 0.8556       |
| **MSCANet**    |  0.8375    |  0.9185  |  0.      |   0.        | 0.        |   0.8599   |
| **DMINet**    |  0.8419      | 0.8669     | 0.     | 0.       |  0.     |  0.8516     |
|   **SARASNet**    |0.8948       |  0.9264   | 0.        |0.         |0.      |  0.8990     |
|  **WNet** |  0.8973     |  0.8991  | 0.       |  0.       |   0.     |  0.8850    |
|  **CSINet** |  0.8861     |  0.9361  | 0.       |  0.       |   0.     |  0.8989    |
| **TTMGNet**     | **0.9316**  |**0.9146** | **0.9231**       |   **0.9832**  | **0.9192**  | **0.9114**|


###  WHU-CD Datasset
#### Qualitative result
<p align="center"> <img src="Fig/road.png" width="90%"> </p>
- Four representative images of the RoadScene test set.In alphabetical order they are infrared image, visible image, GTF, FusionGAN, SDNet, RFN–Nest, U2Fusion, LRRNet, SwinFusion, CDDFuse, DATFuse, and GTMFuse.

#### Quantitative Results

|   **Methods**    |   **EN**   |   **SD**   | **SF** | **VIF** | **AG** | **Qabf** |
|:----------------:|:---------:|:---------:|:---------:|:------------:|:-----------------------:|:---------:|
| **GTF**         |   7.45805    | 10.4952   |   0.04605    |    0.57953     |   4.04458     |   0.37079    |
|  **FusionGAN**  |   7.09511    |   10.0518 |   0.04323    |   0.56307      |     4.11028   |   0.28132    |
| **SDNet**       |   7.3388     | 10.1153   |   0.07541     |   0.74513      |   7.55126    | 0.51691      |
| **RFN–Nest**    |  7.34281    |  10.2000  |  0.05192     |   0.75382       | 5.13552       |    0.45230   |
| **U2Fusion**    |  7.21249      | 10.1205    |  0.07469     |  0.67670       |  7.42630     |  0.51831     |
|   **LRRNet**    |7.09023       |   10.1468  | 0.06907       | 0.64912        | 6.19723      |  0.41013     |
|  **SwinFusion** |  7.18569     |  10.3193  | 0.06757       |  0.80244       |   6.52487     |  0.57124     |
|   **CDDFuse**   |  **7.48812**      |   **10.6921**  |   **0.09099**|     0.78466     |  **8.33022**      |  0.49671      |
| **DATFuse**     |   6.89646   |   10.4078  | 0.05495      |    0.79045     | 5.06397     |  0.50003     |
| **GTMFuse**     | 7.35795  |10.5113| 0.08181      |   **0.87918**  | 7.92432   | **0.60665**  |

### CL-CD Dataset
#### Qualitative result
<p align="center"> <img src="Fig/msrs.png" width="90%"> </p>
- Four representative images of the MSRS test set. In alphabetical order they are infrared image, visible image, GTF, FusionGAN, SDNet, RFN–Nest, U2Fusion, LRRNet, SwinFusion, CDDFuse, DATFuse, and GTMFuse.

#### Quantitative Results

|   **Methods**    |   **EN**   |   **SD**   | **SF** | **VIF** | **AG** | **Qabf** |
|:----------------:|:---------:|:---------:|:---------:|:------------:|:-----------------------:|:-------------------------:|
| **GTF** |   4.44195   | 6.11111  |   0.05620    |    0.48176     |           3.53966            |           0.39194           |
|  **FusionGAN**   |   5.86785   |   6.79263    |   0.03654    |    0.61998      |           3.00051            |           0.24709            |
| **SDNet**  |   5.54468    |   6.13925    |   0.05910    |    0.48644      |        4.40836          |         0.41903           |
| **RFN–Nest**  |   5.81096    |   7.91701    |   0.04982    |   0.74520     |          4.12198           |           0.50474             |
| **U2Fusion**  |  5.03625    | 6.78870  |   0.06157    |   0.57216     |          4.48894            |           0.42512             |
|   **LRRNet**    |5.89799 |   7.30930    | 0.04548  |  0.38422   |          3.64204          |          0.19980           |
|  **SwinFusion**  |  6.61543    |   8.46817   | 0.06756     |  0.99403     |         5.26562           |        0.66481            |
|   **CDDFuse**   |  6.32740   |   8.53021    |   **0.08130**  |     0.97155       |       6.12164           |         0.66558           |
| **DATFuse**  |    6.29844    |   7.71886      | 0.07247  |     0.71196     |       5.96856          |       0.54618          |
| **GTMFuse** |   **6.78256**  |  **8.60603**   | 0.08105  |   **1.00857**   |        **6.39748**          |      **0.69590**        |

If you have any questions, please contact me by email (hux18943@gmail.com).
