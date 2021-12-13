
#!/bin/bash
USERNAME=ubuntu

#LAN->1,2,3
IP1=3.131.77.189			#Ohio
IP2=3.132.42.147  		#Ohio
IP3=3.136.57.237			#Ohio

#WAN->1,4,5
IP4=54.168.225.46     #Tokyo
# IP5=3.9.234.48  			#London
IP5=52.215.29.89  			#Ireland


#########################################################################################
NETWORK=SecureML 			# NETWORK {SecureML, Sarda, MiniONN, LeNet, AlexNet, and VGG16}
DATASET=MNIST 			# DATASET {MNIST, CIFAR10, and ImageNet}
SECURITY=Semi-honest 		# SECURITY {Semi-honest or Malicious} 
RUN_TYPE=WAN 				# RUN_TYPE {LAN or WAN or localhost}
PRINT_TO_FILE=true			# PRINT_TO_FILE {true or false}
FILENAME=time.txt
#########################################################################################

if [[ $PRINT_TO_FILE = true ]]; then
	printf "%s\n" "---------------------------------------------" >> $FILENAME
	printf "%s %s %s %s\n" $RUN_TYPE $NETWORK $DATASET $SECURITY >> $FILENAME
	printf "%s\n" "---------------------------------------------" >> $FILENAME
fi


if [[ $RUN_TYPE = LAN ]]; then
	ssh -i ~/.ssh/falcon-lan1.pem $USERNAME@$IP1 "pkill Falcon.out; echo clean completed; cd falcon-public; rm Falcon.out; make all -j$(nproc); echo maked; chmod +x Falcon.out; ./Falcon.out 0 files/IP_$RUN_TYPE files/keyC files/keyAC files/keyBC $NETWORK $DATASET $SECURITY 1>./time.txt; less time.txt" & 
	ssh -i ~/.ssh/falcon-lan1.pem $USERNAME@$IP2 "pkill Falcon.out; echo clean completed; cd falcon-public; rm Falcon.out; make all -j$(nproc); echo maked; chmod +x Falcon.out; ./Falcon.out 1 files/IP_$RUN_TYPE files/keyB files/keyBC files/keyAB $NETWORK $DATASET $SECURITY 1>./time.txt" & 
	ssh -i ~/.ssh/falcon-lan1.pem $USERNAME@$IP3 "pkill Falcon.out; echo clean completed; cd falcon-public; rm Falcon.out; make all -j$(nproc); echo maked; chmod +x Falcon.out; ./Falcon.out 2 files/IP_$RUN_TYPE files/keyA files/keyAB files/keyAC $NETWORK $DATASET $SECURITY 1>./time.txt" & 
elif [[ $RUN_TYPE = WAN ]]; then
	ssh -i ~/.ssh/falcon-lan1.pem $USERNAME@$IP1 "pkill Falcon.out; echo clean completed; cd falcon-public; rm Falcon.out; make all -j$(nproc); chmod +x Falcon.out; ./Falcon.out 0 files/IP_$RUN_TYPE files/keyA files/keyAB files/keyAC $NETWORK $DATASET $SECURITY 1>./time.txt; less time.txt" & 
	ssh -i ~/.ssh/falcon-tokyo.pem $USERNAME@$IP4 "pkill Falcon.out; echo clean completed; cd falcon-public; rm Falcon.out; make all -j$(nproc); chmod +x Falcon.out; ./Falcon.out 1 files/IP_$RUN_TYPE files/keyA files/keyAB files/keyAC $NETWORK $DATASET $SECURITY 1>./time.txt" & 
	ssh -i ~/.ssh/falcon-ire.pem $USERNAME@$IP5 "pkill Falcon.out; echo clean completed; cd falcon-public; rm Falcon.out; make all -j$(nproc); chmod +x Falcon.out; ./Falcon.out 2 files/IP_$RUN_TYPE files/keyA files/keyAB files/keyAC $NETWORK $DATASET $SECURITY 1>./time.txt" & 
elif [[ $RUN_TYPE = localhost ]]; then
	make all
	./Falcon.out 1 files/IP_$RUN_TYPE files/keyB files/keyBC files/keyAB $NETWORK $DATASET $SECURITY >/dev/null &
	./Falcon.out 2 files/IP_$RUN_TYPE files/keyC files/keyAC files/keyBC $NETWORK $DATASET $SECURITY >/dev/null &
	if [[ $PRINT_TO_FILE = true ]]; then
		./Falcon.out 0 files/IP_$RUN_TYPE files/keyA files/keyAB files/keyAC $NETWORK $DATASET $SECURITY >> $FILENAME
	else
		./Falcon.out 0 files/IP_$RUN_TYPE files/keyA files/keyAB files/keyAC $NETWORK $DATASET $SECURITY 
	fi
else
	echo "RUN_TYPE error" 
fi




########################################## SET-UP COMMANDS ##########################################
#sudo apt-get update; sudo apt-get install g++; sudo apt-get install libssl-dev; sudo apt install make; sudo apt-get install iperf3 
#git clone https://github.com/snwagh/falcon-public.git; cd falcon-public

# ssh -i ~/.ssh/falcon_sp_oregon.pem ubuntu@18.237.39.209
# ssh -i ~/.ssh/falcon_sp_oregon.pem ubuntu@34.221.35.166
# ssh -i ~/.ssh/falcon_sp_oregon.pem ubuntu@34.219.97.126
