# Charuco coordinates transformation
Setup:  
* FLOOR_TARGET is a charuco board, mounted on the floor.  
* TRACKING_CAMERA have a wide-angle top-down view of the entire scene and is used to properly track the position of the FILM_TARGTET.  
* FILM_CAMERA have a FILM_TARGTET charuco board, mounted on the top of camera.
## Stage 1
At the start, we know all cameras position in the FLOOR_TARGET coordinarte system.  
We need to calculate the FILM_VECTORS, which are translation and rotation vectors from FILM_CAMERA to FILM_TARGET that will be used further.<br>
<div align="center">
<img src="./assets/scheme_no_obstacle.png"><br>
<table style="border: none;">
    <tr>
        <th style="border: none;">Tracking camera</th>
        <th style="border: none;">Film camera</th>
    </tr>
    <tr>
        <td style="border: none;"><img src="./renders/0/TrackingCameraView.png" width="300"></td>
        <td style="border: none;"><img src="./renders/0/FilmCameraView.png" width="300"></td>
    </tr>
</table>
</div>

## Stage 2
Even as the FILM_CAMERAs begin to move and rotate, our objective remains to accurately track the position and rotation of the FILM_CAMERA in the FLOOR_TARGET coordinate system, regardless of its visibility. To achieve this, we need to establish a chain of reference:  
FLOOR_TARGET -> TRACKING_CAMERA -> FILM_TARGET -> FILM_VECTORS  
Consequently, it is crucial to devise a method that allows us to compute the FILM_CAMERA's position and rotation vectors in the FLOOR_TARGET coordinate system, even when the FLOOR_TARGET is out of sight.  
![Production stage](./assets/scheme_obstackle.png)<br>
## What we have
...