# Download dataset from Dropbox
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1KYKMajoyan6Dk5rZjAHqSOGRDCbHKMCj' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1KYKMajoyan6Dk5rZjAHqSOGRDCbHKMCj" -O digit_data.zip && rm -rf /tmp/cookies.txt

# Unzip the downloaded zip file
unzip ./digit_data.zip

# Remove the downloaded zip file
rm ./digit_data.zip
