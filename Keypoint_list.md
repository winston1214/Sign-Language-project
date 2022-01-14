## Keypoint List

We extracted the keypoint based on <a href='https://github.com/Fang-Haoshu/Halpe-FullBody'>Halpe 136</a> and went through some modifications.

Since the lower body did not appear due to the nature of the dataset, the key point for the lower body was removed.

### Keypoint Matrix

|index|point|
|----|----|
|0|Nose|
|1|LEye|
|2|REye|
|3|LEar|
|4|REar|
|5|LShoulder|
|6|RShoulder|
|7|LElbow|
|8|RElbow|
|9|LWrist|
|10|RWrist|
|11|Head|
|12|Neck|
|13~80|Face|
|81~101|LHand|
|102~123|RHand|

<p align="center">
  <img src='https://github.com/Fang-Haoshu/Halpe-FullBody/raw/master/docs/human_model.jpg' width="30%" height="30%"></img>
</p>
<p align="center">
  Body Keypoint(remove 11,12,13,14,15,16,19,20,21,22,23,24,25)
</p>

<p align="center">
  <img src='https://github.com/Fang-Haoshu/Halpe-FullBody/raw/master/docs/face.jpg' width="50%" height="50%"></img>
</p>
<p align="center">
  Face Keypoint
</p>

<p align="center">
  <img src="https://github.com/Fang-Haoshu/Halpe-FullBody/raw/master/docs/hand.jpg"></img>
</p>
<p align='center'>
  Hand Keypoint
</p>
