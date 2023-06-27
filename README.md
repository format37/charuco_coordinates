# Charuco coordinates transformation
Setup:  
* FLOOR_TARGET is a charuco board, mounted on the floor.  
* TRACKING_CAMERA have a wide-angle top-down view of the entire scene and is used to properly track the position of the FILM_TARGTET.  
* FILM_CAMERA have a FILM_TARGTET charuco board, mounted on the top of camera.  
  
At the start, we know all cameras position in the FLOOR_TARGET coordinarte system.  
  
But then the film cameras starting to move and rotate.  
FILM_CAMERA will not always will see tha FLOOR_TARGET, but we still need to know the FILM_CAMERA position and rotation in the FLOOR_TARGET coordinate system.  
  
We need to restore FILM_CAMERA position by the chain:  
FLOOR_TARGET -> TRACKING_CAMERA -> FILM_TARGET -> FILM_VECTORS.

## Stage 1
First, we have a calibration stage, when all cameras can see the FLOOR_TARGET which is defines the absolute 0,0,0 point of the scene.<br>
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
Next, even if the FLOOR_TARGET is not visible, we still need to be able to calculate the FILM_CAMERA position and rotation vectors in the FLOOR_TARGET coordinate system. In addition, we need to account that the FILM_CAMERA + FILM_TARGET has changed its position and rotation.  
![Production stage](./assets/scheme_obstackle.png)<br>
According camera views is:<br>

## What we have
...