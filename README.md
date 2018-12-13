# General Edge
This is a general framework for Edge Computing on Raspberry Pis. All modules are written in Python3. 

### Table of Contents
- [Outline](#outline)
  * [Pi Manager](pi-manager)
  * [Config](config)
  * [Data Collector](data-collector)
  * [Algorithm](algorithm)
- [Get started](#get-started)
  * [Wifi and SSH Configuration](#wifi-and-ssh-configuration)
  * [Install](#install)
  * [Raspberry Pi Configuration](#raspberry-pi-configuration)
  * [Run](#run)
  
### Outline

#### Pi Manager
The <b>pi_manager.py</b> is the manager of the whole project, it will read the config file, learn the rules about this experiment, then use corresponding strategy to do data collection or run the algorithm.
#### Config
The config file contains the rules of the whole experiment, the general structure looks like:
```yaml
# name of the pi, you should specify this when you run the program.
sensorPiA
    # What should this Pi do.
    role:Algorithm/DataCollector
    # Path of the corresponding class you want to use.
    classPath: algorithm/data_collector
    # The corresponding class.
    className: LinearRegression
    # Bluetooth address of this Pi.
    btAddress: XX:XX:XX:XX:XX:XX
    # The next handler inside the yaml file
    downStream:
    # room in DynamoDB(optional)
    room: roomA
    # sensor in DynamoDB(optional)
    sensor: sensorA
    # Path to your data(optional for data collector)
    dataFilePaths:
        - data/dataA
```
#### Data collector
The data collector will work like the sensor, it collects data and send it to its downstream. 
```
def init() //Initialize this data collector, pass in necessary parameters.
def cleanup() //Clean this data collector.
def send() //Send data to the downstream.
def run() //Get data.
```

#### Algorithm
The algorithm module should contains all the logic you want your gateway to do. Generally, you should implement different ways to handle local computing and transfer data to downstream directly.
```
def init() //Initialize this algorithm part, pass in necessary parameters.
def run() //Run the algorithm module.
def cleanup() //Clean this algorithm module.
def send() //Send data to the downstream.
```

### Get started

#### Local Pipeline

##### Wifi and SSH Configuration
If you want Raspberry Pis to connect Wifi and use SSH to login the Pis, you should put <b>utils/ssh</b> and <b>utils/wpa_supplicant.conf</b> in the root directory of your Raspberry Pi.

##### Configure AWS credential
```
export aws_access_key_id="XXXXXX"
export aws_secret_access_key="XXXXX"
```

##### Install
On every Raspberry Pi, clone this repository.
```sh
$ git clone https://github.com/YoungYang0820/General_Edge.git
```

##### Raspberry Pi Configuration
On every Raspberry Pi, you need to open bluetooth at first.
```sh
$ sudo apt-get install bluetooth libbluetooth-dev
$ sudo python3 -m pip install pybluez

$ sudo /etc/init.d/bluetooth restart
$ sudo bluetoothctl
// Inside the bluetoothctl
    $ scan on
    $ power on
    $ agent on
    $ discoverable on
```

Then, install the corresponding packages.
```sh
sudo pip3 install -r requirement.txt
```

##### Run
You should run based on your config file. 
```sh
$ python3 pi_manager.py [config file] [pi name]

$ python3 pi_manager.py base_cfg.yaml sensorPiA
$ python3 pi_manager.py base_cfg.yaml gatewayPiA
```

#### Remote Lambda

##### Package
* Enter the lambda_functions/ and use `make` command to generate required files.
```sh
make mnist_nn
```
* `scp` all files inside the lambda_functions/package/ to remote `ec2 VM`
```sh
scp lambda_functions/package/* remote@vm:~/your_own_path
```
* Install all required **python-packages** in `ec2 VM`

* Generate zip file and upload it to `s3` bucket
https://docs.aws.amazon.com/lambda/latest/dg/lambda-python-how-to-create-deployment-package.html

##### Lambda Function Configurations
* Memory Limit -> 1024MB
* Timeout Limit -> 10 min
* Environment Variables:
```
aws_access_key_id="XXXXXX"
aws_secret_access_key="XXXXX"
```
