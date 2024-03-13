# DeepAC (inference) pseudo-code

(`src_open/tools/demo.py`)
1. Load the pre-trained model
2. Load 3D model data (viewpoints, 3D points and normals, ...)
3. Load image sequence data (images, ground truth poses, camera intrinsic parameters) and initialize the pose with ground truth
4. For each image of the sequence:
   1. Find the k closest views to the current pose (to avoid working with the whole set of viewpoints)
   2. Get the correspondence lines in the image plane by projecting the 3D points and normals of the closest view
   3. Find the bounding box of the object in the image from the correspondence lines' centers and extract it
   4. Initialize the histograms with the **first** cropped image of the sequence
   5. Gather the input data for the model (cropped image, camera parameters, initial pose, k closest views, k closest orientations, histograms)
   6. Pass the input data through the model to get the predicted pose:
        (`src_open/models/deepac.py`)
      1. Feature maps extraction
      2. For each scale (coarse-to-fine):
         1. Downsample the RGB cropped image and select the feature map of the current scale
         2. Extract the correspondence lines (positions, normals) for the current scale
         3. Extract the contour feature map (grid sampling)
         4. Predict the boundary probability map
         5. Compute the gradient and the Hessian matrix of the joint posterior probability (boundary probability map)
         6. 1 step of the Newton method to update the pose
         7. Update the variable which stores the predicted pose
   7. Update the initial pose for the next frame with the predicted pose
   8. Online adaptation of the histograms with the current cropped image and the predicted pose
