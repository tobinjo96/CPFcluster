#Data sets 

The data sets used in the experimental section of our work are available at the following links: 
* [Cov Type](https://drive.google.com/file/d/1ncsZtAy1Q3rWAEyztme4eO37f1JEdUU6/view?usp=sharing) (58MB) - UCI Machine Learning Repository.
* [KDD Cup '99](https://drive.google.com/file/d/11ql7MDPBsutV3Rv8AD2vVSP-9_H-IG_H/view?usp=sharing) (2GB) - UCI Machine Learning Repository. Column Index: [0,0,0
* [YouTube Faces](https://drive.google.com/file/d/11S7pIp9uDde-AG9c5gAXYwhl0DzDeUWn/view?usp=sharing) (304MB) - Lior Wolf, Tal Hassner and Itay Maoz. 'Face Recognition in Unconstrained Videos with Matched Background Similarity.' Details [here](https://www.cs.tau.ac.il/~wolf/ytfaces/). 
* [MS-Celeb-1M](https://drive.google.com/file/d/1ZBp5fu-PGiNBSnHIf5kV7XRaea2ujdVC/view?usp=sharing) (2GB) - Yandong Guo, Lei Zhang, Yuxiao Hu, Xiaodong He. 'MS-Celeb-1M: Challenge of Recognizing One Million Celebrities in the Real World' Details [here](https://www.microsoft.com/en-us/research/project/ms-celeb-1m-challenge-recognizing-one-million-celebrities-real-world/). 

Each of the data files is a .npy file containing the data array. Categories have been integer encoded for compatibility with numpy. For each of these data sets, the label vector is the last column in the array. 

The column indices for each data set are:
* Cov Type - [0,0,0,0,0,0,0,0,0,0,1,1,2]
* KDD Cup '99 - [0,1,1,1,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2]
* YouTube Faces - [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2]
* MS-Celeb-1M - [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2]

