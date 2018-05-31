## Presidential RNN
This respository trains and executes a character-level recurrent neural network using GRUs on 
President Donald Trump's tweets and official recorded statements and speeches.  Equivalent Markov 
chains are also trained, to limited effect.

### Running the Repository
The primary entry points for this repository are `main.py` and `markov.py`.  They may be accessed
by the command `python [filename]`.

Running `main.py` as-is will load and transform both tweets and statements, training 6 and dumping
RNN models each for a total of 12 models.  This is extraordinarily computationally expensive.

Running `markov.py` will generate frequency tables and sentences given a seed.  Frequency table 
generation is an expensive task, and may take hours.

Pre-trained models are stored in `/data/models/` in `h5py` format and may be used directly. 

#### Full Dependencies
Core Functionality
* `keras`
* `sklearn`
* `pandas`
* `numpy`
* `unidecode`

EDA Functionality (topic modeling, visualizations, etc.)
* `gensim`
* `yellowbrick`

### Repository Structure
* `bin` - Source code files
  * `pytorch` - pyTorch implementation of GRU, unoptimized for CUDA or GPU
* `data`
  * `clean` - Cleaned data, stored in serialized format
  * `models` - Trained char-RNNs, stored in `h5py` format
  * `markov` - Markov chain frequency tables and sample output
* `docs` - Slides and supporting documentation for project

### Contact
Feel free to contact me with feedback or questions.  
 `Email` - `lzhou95` at `gmail` .com  
`LinkedIn` - [zhouleon](https://www.linkedin.com/in/zhouleon/)  
`Medium` - [@confusionmatrix](https://medium.com/@confusionmatrix)
