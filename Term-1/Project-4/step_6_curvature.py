import numpy as np
import cv2

def calculate_radius(left_fit, right_fit):
    
#    y_eval = np.max(ploty)
    y_eval = 719
    left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
    right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
#    print(left_curverad, right_curverad)
    txt = "radius left/right lane (m): " + str(int(left_curverad)) + "/" + str(int(left_curverad))
    return txt

def test(left_fit, right_fit):
   calculate_radius(left_fit, right_fit) 
   
   
def write_text_on_image(image, text):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image,text,(10,50), font, 1,(255,255,255),2)
    return image



#test(polinomials["left_fit"], polinomials["right_fit"])    
#    
 