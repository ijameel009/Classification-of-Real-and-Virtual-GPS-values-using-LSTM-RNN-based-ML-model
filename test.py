import airsim
import pprint
import cv2
import numpy as np

#clients for UAV to be able make connection with unreal engine and AirSim
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)

#client to be sync with while takeoff
client.armDisarm(True)
client.takeoffAsync().join()

#new positions of UAV after its moved

x = 0
y = 0
z = -5
#prev position of UAV

prevX = -1
prevY = -1
prevZ = -1

# header = "lon,lat,alt
file = open("data.csv", "w")   #to form a dataset for theGPS valaues
header = "lon,lat,alt"
file.write(header + "\n")
while True:
    i = np.zeros((200, 200)).astype("uint8")
    gps_data = client.getGpsData()
    gps_str = pprint.pformat(gps_data)
    if prevX != x or prevY != y or prevZ != z:
        client.moveToPositionAsync(x, y, z, 5).join()
    print("\nGPS Data")
    print(type(gps_str))
    data_str = f"{gps_data.gnss.geo_point.longitude},{gps_data.gnss.geo_point.latitude},{gps_data.gnss.geo_point.altitude}, "
    file.write(data_str + "\n")
    print(gps_data)
    prevX = x
    prevY = y
    prevZ = z

    cv2.imshow("KeyBoard", i)
    k = cv2.waitKey(30)

    if k == ord("w"):   #Move Forward
        x += 2
    if k == ord("s"):   #Move Backward
        x -= 2

    if k == ord("a"):   #Move Left
        y += 2
    if k == ord("d"):   #Move Right
        y -= 2

    if k == ord("r"):   #Move Downward
        z += 2
    if k == ord("f"):   #Move Upward
        z -= 2
    if k == ord("q"):   #Recording
        break
file.close()
client.reset()
client.armDisarm(False)

# that's enough fun for now. let's quit cleanly
client.enableApiControl(False)