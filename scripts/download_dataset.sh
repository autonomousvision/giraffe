echo "This script downloads datasets used in the GIRAFFE project."
echo "Choose from the following options:"
echo "0 - Cats Dataset"
echo "1 - CelebA Dataset"
echo "2 - Cars Dataset"
echo "3 - Chairs Dataset"
echo "4 - Church Dataset"
echo "5 - CelebA-HQ Dataset"
echo "6 - FFHQ Dataset"
echo "7 - Clevr2 Dataset"
echo "8 - Clevr2345 Dataset"
read -p "Enter dataset ID you want to download: " ds_id

if [ $ds_id == 0 ]
then
    echo "You chose 0: Cats Dataset"
    mkdir -p data
    cd data
    echo "Start downloading ..."
    wget https://s3.eu-central-1.amazonaws.com/avg-projects/giraffe/data/cats.zip
    echo "done! Start unzipping ..."
    unzip cats.zip
    echo "done!"
elif [ $ds_id == 1 ]
then
    echo "You chose 1: CelebA Dataset"
    mkdir -p data
    cd data
    echo "Start downloading ..."
    wget https://s3.eu-central-1.amazonaws.com/avg-projects/giraffe/data/celeba.zip
    echo "done! Start unzipping ..."
    unzip celeba.zip
    echo "done!"
elif [ $ds_id == 2 ]
then
    echo "You chose 2: Cars Dataset"
    mkdir -p data
    cd data
    echo "Start downloading ..."
    wget https://s3.eu-central-1.amazonaws.com/avg-projects/giraffe/data/comprehensive_cars.zip
    echo "done! Start unzipping ..."
    unzip comprehensive_cars.zip
    echo "done!"
elif [ $ds_id == 3 ]
then
    echo "You chose 3: Chairs Dataset"
    mkdir -p data
    cd data
    echo "Start downloading ..."
    wget https://s3.eu-central-1.amazonaws.com/avg-projects/giraffe/data/chairs.zip
    echo "done! Start unzipping ..."
    unzip chairs.zip
    echo "done!"
elif [ $ds_id == 4 ]
then
    echo "You chose 4: Church Dataset"
    echo "Note: We only provide our ground truth FID activations. Please checkout the original dataset for the image data."
    mkdir -p data
    cd data
    echo "Start downloading ..."
    wget https://s3.eu-central-1.amazonaws.com/avg-projects/giraffe/data/church.zip
    echo "done! Start unzipping ..."
    unzip church.zip
    echo "done!"
elif [ $ds_id == 5 ]
then
    echo "You chose 5: CelebA-HQ Dataset"
    echo "Note: We only provide our ground truth FID activations. Please checkout the original dataset for the image data."
    mkdir -p data
    cd data
    echo "Start downloading ..."
    wget https://s3.eu-central-1.amazonaws.com/avg-projects/giraffe/data/celebahq.zip
    echo "done! Start unzipping ..."
    unzip celebahq.zip
    echo "done!"
elif [ $ds_id == 6 ]
then
    echo "You chose 6: FFHQ Dataset"
    echo "Note: We only provide our ground truth FID activations. Please checkout the original dataset for the image data."
    mkdir -p data
    cd data
    echo "Start downloading ..."
    wget https://s3.eu-central-1.amazonaws.com/avg-projects/giraffe/data/ffhq.zip
    echo "done! Start unzipping ..."
    unzip ffhq.zip
    echo "done!"
elif [ $ds_id == 7 ]
then
    echo "You chose 7: Clevr2 Dataset"
    mkdir -p data
    cd data
    echo "Start downloading ..."
    wget https://s3.eu-central-1.amazonaws.com/avg-projects/giraffe/data/clevr2.zip
    echo "done! Start unzipping ..."
    unzip clevr2.zip
    echo "done!"
elif [ $ds_id == 8 ]
then
    echo "You chose 8: Clevr2345 Dataset"
    mkdir -p data
    cd data
    echo "Start downloading ..."
    wget https://s3.eu-central-1.amazonaws.com/avg-projects/giraffe/data/clevr2345.zip
    echo "done! Start unzipping ..."
    unzip clevr2345.zip
    echo "done!"
else
    echo "You entered an invalid ID!"
fi
