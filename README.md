# Boundary-Growing-in-Python
Body Part Detection using Haar Cascade Detector and Boundary Growing Algorithm

Here in the code those who want to Use this code can modify the face_cascade.detectMultiScale3 parameters accordingly , minimum size of window is 400 and max size i 500 and threshold is 1.02 , anyone can modify this and use it 

Algorithm is Simple:

1. Take a input image(undetected/not good detection image)
2. Feed the classifier , it will classify and produce a list of Probable faces( here you can add body , skin etc but for that change threshold and classifier ) 
3. For each face it will check the confidence score of the HAAR Cascade Classifier , once it is monotonic Increasing it means it is correctly detecting face 
4. For each iteration for a face , i am checking if the confidence score is increasing or not ? if increasing , i am expanding the detector window size by a constant ( here 20 pixels) 
5. After increasing monotonically , i will check when my confidence score decreases , that peak point will be the point of stopping iteration
6. The resulting image will be the correctly detected face ( 100 % accuracy )
