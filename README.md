# MineRL


### Installation

Following the instructions in MineRL did not work for us. To run the project, first:
1.  Install Mine RL package and dependencies

```
pipenv install
```

2. Install and work on a good version of java to avoid Malmo errors.
```
sudo apt-get purge openjdk-8*
sudo apt-get install openjdk-8-jre-headless=8u162-b12-1
sudo apt-get install openjdk-8-jdk-headless=8u162-b12-1
sudo apt-get install openjdk-8-jre=8u162-b12-1
sudo apt-get install openjdk-8-jdk=8u162-b12-1
sudo update-java-alternatives -s java-1.8.0-openjdk-amd64
```

3. Run the code.

```
TMP: 
pipenv shell
python navigate.py
```