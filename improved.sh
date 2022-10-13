# TODO: create shell script for running your improved UDA model
#!/bin/bash
if [[ "$2" == "mnistm" ]]; then
    echo "testing mnistm ..."
    checkpoint="P4_mnistm_model.pkl"
    wget -O ${checkpoint} 'https://www.dropbox.com/s/lofx8agzlzq9bu3/P4_usps-mnistm_last_checkpoint.pkl?dl=1'
    python3 P4_main.py --phase test_da --checkpoint ${checkpoint} --test_dir $1 --test_dataset $2 --out_csv $3
elif [[ "$2" == "usps" ]]; then
    echo "testing usps ..."
    checkpoint="P4_usps_model.pkl"
    wget -O ${checkpoint} 'https://www.dropbox.com/s/k3utue741nixtwg/P4_svhn-usps_last_checkpoint.pkl?dl=1'
    python3 P4_main.py --phase test_da --checkpoint ${checkpoint} --test_dir $1 --test_dataset $2 --out_csv $3
else # svhn
    echo "testing svhn ..."
    checkpoint="P4_svhn_model.pkl"
    wget -O ${checkpoint} 'https://www.dropbox.com/s/kk452wtucvqzkuk/P4_mnistm-svhn_last_checkpoint.pkl?dl=1'
    python3 P4_main.py --phase test_da --checkpoint ${checkpoint} --test_dir $1 --test_dataset $2 --out_csv $3
fi
