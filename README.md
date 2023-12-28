# U-Net: Semantic segmentation

- Base Official Algorithm: [U-Net](https://arxiv.org/abs/1505.04597)

- Base Github project: [milesial/Pytorch-UNet](https://github.com/milesial/Pytorch-UNet)

## Model Description
- UNet은 Encoder-decoder architecture 모델로 의료 영상 분석에서 우수한 성적을 냈으며 이와 특성이 비슷한 고해상도 위성 이미지 데이터에서도 좋은 성능이 증명됨
- 고해상도 이미지를 Patch 단위로 잘라 분석하며 Skip connection과 Dense convolution block 구조를 활용
- 해당 모델은 Overlab-tile strategy를 활용해 큰 이미지를 겹치는 부분이 있도록 나누고 모델의 입력으로 활용
- Weight Loss를 활용하여 모델이 객체간 경계를 잘 구분할 수 있도록 Loss를 설정함

## Model Architecture
![network architecture](https://i.imgur.com/jeDVpqF.png)

### Encoder
- Encoder 또는 축소경로(Contracting Path)라고 하며, 각 단계 마다 Decoder 단계로 복사하기 위한 Double Convolution Block과 차원을 축소하여 Encoder의 다음 단계로 보내기 위한 Down Sampling Block으로 나뉨
- n_channels는 모델의 input image의 채널 수로, 본 프로젝트에서는 3채널(RGB) 이미지를 사용하기 때문에 3을 입력

```
  ...

  self.inc = DoubleConv(n_channels, 64)
  self.down1 = Down(64, 128)
  self.down2 = Down(128, 256)
  self.down3 = Down(256, 512)
  factor = 2 if bilinear else 1
  self.down4 = Down(512, 1024 // factor)  ### 이 단계는 bridge block이라고 구분하기도 함

  ...
```
#### Double Convolution Block
- Convolution, Batch Normalization, ReLU 활성화 함수로 이루어짐
```
  ...

  self.double_conv = nn.Sequential(
    nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
    nn.BatchNorm2d(mid_channels),
    nn.ReLU(inplace=True),
    nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
    nn.BatchNorm2d(out_channels),
    nn.ReLU(inplace=True)
  )

  ...
```

#### Down Sampling Block
- 2x2 max pooling으로 Down Sampling 하여 Encoder의 다음 단계로 보내고, Double Convolution Block을 여기서 호출하여 Decoder 단계로 복사함
```
  ...

  self.maxpool_conv = nn.Sequential(
    nn.MaxPool2d(2),
    DoubleConv(in_channels, out_channels)
  )

  ...
```
### Decoder
- Decoder 또는 확대경로(Expanding Path)라고 하며, Encoder 단계에서 복사된 대칭되는 맵을 출력하는 Double Convolution Block과 이전 단계에서 넘어온 feature map의 해상도는 2배로, 채널 수는 절반으로 줄이는 Transposed Convolution(또는 Normal Convolution) Block으로 이루어짐 
```
  self.up1 = Up(1024, 512 // factor, bilinear)
  self.up2 = Up(512, 256 // factor, bilinear)
  self.up3 = Up(256, 128 // factor, bilinear)
  self.up4 = Up(128, 64, bilinear)
  self.outc = OutConv(64, n_classes)
```
#### Up Sampling Block
- Normal Convolution 또는 Transposed Convolution으로 Up Sampling 하여 Decoder의 다음 단게로 보내고, Double Convolution Block을 호출하여 Encoder에서 복사된 맵을 출력함
```
  ...

  # if bilinear, use the normal convolutions to reduce the number of channels
  if bilinear:
      self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
      self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
  else:
      self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
      self.conv = DoubleConv(in_channels, out_channels)

  ...
```

## Input Data

#### 항공영상
- 1024 * 1024 * 3 크기의 tif 포맷 이미지

![항공영상 input](./images/AP_HR_2021_0172_07_input.png)


#### 위성이미지
- 128 * 128 * 3 크기의 tif 포맷 이미지

![위성이미지 input](./images/OS_HR_20210303_0225_input.png)

#### 위성SAR
- 128 * 128 * 3 크기의 tif 포맷 이미지

![위성SAR input](./images/SS_HR_20210305_0236_input.png)

## Output Data

#### 항공영상
- 1024 * 1024 * 3 크기의 tif 포맷 이미지

![항공영상 output](./images/AP_HR_2021_0172_07_output.png)

#### 위성이미지
- 128 * 128 * 3 크기의 tif 포맷 이미지

![위성이미지 output](./images/OS_HR_20210303_0225_output.png)

#### 위성SAR
- 128 * 128 * 3 크기의 tif 포맷 이미지

![위성SAR output](./images/SS_HR_20210305_0236_output.png)

## Training Dataset

- 1024 * 1024 크기의 tif 포맷 항공 영상 및 6개 클래스에 대한 segmentation이 포함된 geojson 24,000 쌍 이상
- 128 * 128 크기의 tif 포맷 항공 영상 및 1개 클래스에 대한 segmentation이 포함된 geojson 24,000 쌍 이상
- 128 * 128 크기의 tif 포맷 항공 영상 및 1개 클래스에 대한 segmentation이 포함된 geojson 12,000 쌍 이상


## Training Parameters

- batch_size = 8
- epochs for 항공영상 = 30
- epochs for 위성이미지 & 위성SAR = 50
- Training:Valid:Test = 8:1:1

## Evaluation Metric

- Overall Pixel Accuracy
- 올바르게 분류한 픽셀 수(True positives, n_ii)를 i개의 클래스로 구성된 전체 이미지 픽셀 수로 나눈 값(t_i)

![Overall Pixel Accuracy](images/eval_metric.png)

