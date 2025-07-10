# Covid-19-ML
The proposed system will be classifying the chest x-ray images (CXR) into covid or
normal CXR. This classification can be helpful in isolating the covid-19 affected
patient immediately. The classification is based on finding the presence of glassy
formation in the CXR images. Finding this change in the CXR images is challenging
even for the professionals in the naked eye. So they prescribe to take CT-scan to
visualize the Chest region a lot clearly. But the motivation is to minimize the cost of
the covid test. CT scan costs more comparatively with the CXR imaging. In recent
years CNN plays a vital role in image classification and object detection. There are
various CNN models which are performing for various purposes. The proposed
system will be using Xception model for feature extraction. The extracted feature is
then classified with the help of Logistic regression.

1.User Interface:
![image](https://github.com/user-attachments/assets/21d8c51e-2223-4cad-a185-4b043ab27e2b)

2.OUTPUT:
When the predict button is pressed, the output will be
![image](https://github.com/user-attachments/assets/b189812c-b88e-41a8-8df1-06515962aaf8)


3.Data Visualization:
The vision of the deep learning model can be visualized with the help of
the grad-cam package. Grad-CAM uses the gradients of any target concept, flowing
into the final convolutional layer to produce a coarse localization map highlighting the
important regions in the image for predicting the concept.

Normal X-ray image

![image](https://github.com/user-attachments/assets/5e4a4495-9ad3-40e2-9805-56dcffa40972)

Covid X-ray image

![image](https://github.com/user-attachments/assets/d34b6294-5e97-42ac-9a0e-fdb8f87f1a7d)
