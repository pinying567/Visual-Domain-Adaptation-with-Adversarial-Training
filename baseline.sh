# TODO: create shell script for running your DANN model
#!/bin/bash
if [[ "$2" == "mnistm" ]]; then
    echo "testing mnistm ..."
    checkpoint="P3_mnistm_model.pkl"
    wget -O ${checkpoint} 'https://www.dropbox.com/s/xut3hrd1l1p2dap/P3_usps-mnistm_last_checkpoint.pkl?dl=1'
    python3 P3_main.py --phase test --checkpoint ${checkpoint} --test_dir $1 --test_dataset $2 --out_csv $3
elif [[ "$2" == "usps" ]]; then
    echo "testing usps ..."
    checkpoint="P3_usps_model.pkl"
    wget -O ${checkpoint} 'https://www.dropbox.com/s/zenbd245ray8q9k/P3_svhn-usps_last_checkpoint.pkl?dl=1'
    python3 P3_main.py --phase test --checkpoint ${checkpoint} --test_dir $1 --test_dataset $2 --out_csv $3
else # svhn
    echo "testing svhn ..."
    checkpoint="P3_svhn_model.pkl"
    wget -O ${checkpoint} 'https://www.dropbox.com/s/jrpzumh454iruw7/P3_mnistm-svhn_last_checkpoint.pkl?dl=1'
    python3 P3_main.py --phase test --checkpoint ${checkpoint} --test_dir $1 --test_dataset $2 --out_csv $3
fi
