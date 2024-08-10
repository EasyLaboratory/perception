# Pure Vision Is All You Need
## usage
1.Create the workspace and clone the perception package
``` bash
mkdir easy_ws
cd easy_ws
mkdir src
cd src
git clone https://github.com/EasyLaboratory/perception.git
cd perception
```
2.Set up the virtual environment
```bash
pip -m venv yolo_venv
source yolo_venv/bin/activate
pip install -r requirements.txt
```
3.Start the airsim ros node, referring to [airsim official page](https://microsoft.github.io/AirSim/airsim_ros_pkgs/]).
4.Copy the settings file

5.Launch the perception node.If you use remote connection(WSL works the same), set the remote_ip to the ip of your remote airsim simulator. And the remote_ip default value is 127.0.0.1 which will work when the airsim simulator works in the same machine.
```bash
roslaunch perception perception.launch remote_ip:=192.168.1.12
``` 
## Tips
1. **Set the intrinsic parameter properly when the camera setting changes.** The intrinsic value is set in K as shown below.
```python
K=[320.0, 0.0, 320.0, 0.0, 320.0, 240.0, 0.0, 0.0, 1.0]
``` 
You can look at the K in airsim topic and copy the K value to the source code.
```bash
# get the specific camera topic
rostopic list
rostopic echo /some/camera/info
```
## Road Map
1. Improve the model accuracy to adapt the changing road conditions.
2. Process the noise of the deep camera.
3. Process the target loss condition.
4. Deploy the model to a more general condition, such as use quantitative pruning and distillation to accelerate the inference process.
5. Rewrite this piece of code to c++. 


