# CPFcluster

CPFcluster is a Python library for implementing the Component-wise Peak Finding (CPF) method introduced in 'Scalable Clustering of Mixed Data using Nearest Neighbor Graphs and Density Peaks'. 

## Set Up

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install foobar.

```bash
pip install CPFcluster
```

## Run

To run CPFcluster on the synthetic data included:

```bash
python CPF_synthetic.py 
```

To run CPFcluster on the downloaded data sets, see the Dataset.md file in the ./data folder. The available data sets can be found there. They should be downloaded to the ./data folder. The below command will execute CPFcluster on the data sets. 

```bash
python CPF_downloaded.py
```
## License
[MIT](https://choosealicense.com/licenses/mit/)

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## Questions or Comments
Please contact Joshua Tobin ([tobinjo@tcd.ie](mailto:tobinjo@tcd.ie)). 

Future additions to the repository will provide ways to pass arguments to CPF from the command line. 