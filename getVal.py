import cv2

cap = cv2.VideoCapture(0)

fps = cap.get(cv2.CAP_PROP_FPS)
print("Frames per second using video.get(cv2.CAP_PROP_FPS): {0}".format(fps))

gain = cap.get(cv2.CAP_PROP_GAIN)
print("Frames per second using video.get(cv2.CAP_PROP_GAIN): {0}".format(gain))

saturation = cap.get(cv2.CAP_PROP_SATURATION)
print("Frames per second using video.get(cv2.CAP_PROP_Saturation): {0}".format(saturation))

hue = cap.get(cv2.CAP_PROP_HUE)
print("Frames per second using video.get(cv2.CAP_PROP_HUE): {0}".format(hue))

contrast = cap.get(cv2.CAP_PROP_CONTRAST)
print("Frames per second using video.get(cv2.CAP_PROP_CONTRAST): {0}".format(contrast))

brightness = cap.get(cv2.CAP_PROP_BRIGHTNESS)
print("Frames per second using video.get(cv2.CAP_PROP_BRIGHTNESS): {0}".format(brightness))

sharpness = cap.get(cv2.CAP_PROP_SHARPNESS)
print("Frames per second using video.get(cv2.CAP_PROP_SHARPNESS): {0}".format(sharpness))

auto_exposure = cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)
print("Frames per second using video.get(cv2.CAP_PROP_AUTO_EXPOSURE): {0}".format(auto_exposure))

exposure = cap.get(cv2.CAP_PROP_EXPOSURE)
print("Frames per second using video.get(cv2.CAP_PROP_EXPOSURE): {0}".format(exposure))

autofocus = cap.get(cv2.CAP_PROP_AUTOFOCUS)
print("Frames per second using video.get(cv2.CAP_PROP_AUTOFOCUS): {0}".format(autofocus))

backlight = cap.get(cv2.CAP_PROP_BACKLIGHT)
print("Frames per second using video.get(cv2.CAP_PROP_BACKLIGHT): {0}".format(backlight))

focus = cap.get(cv2.CAP_PROP_FOCUS)
print("Frames per second using video.get(cv2.CAP_PROP_FOCUS): {0}".format(focus))

frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
print("Frames per second using video.get(cv2.CAP_PROP_FRAME_COUNT): {0}".format(frame_count))

trigger = cap.get(cv2.CAP_PROP_TRIGGER)
print("Frames per second using video.get(cv2.CAP_PROP_TRIGGER): {0}".format(trigger))

speed = cap.get(cv2.CAP_PROP_SPEED)
print("Frames per second using video.get(cv2.CAP_PROP_SPEED): {0}".format(speed))



