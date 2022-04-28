# GraphFlow

This code is forked from [GraphFlow](https://github.com/hugochan/GraphFlow) repository. Changes are made in repository to succesfully run the results and some engineering tricks to achieve better performance.


### Run the model

* Download the preprocessed data from [here](https://1drv.ms/u/s!AjiSpuwVTt09gTtAGzIRsp6Py3q-?e=Yxqa7w) and put the data folder under the root directory.
Download [300-dim 830B GloVe embeddings](https://github.com/stanfordnlp/GloVe).
* Tried models with lesser size embeddings but does not perform better than above given embedding. However, larger size embeddings to give better performance but is computational expensive too! 
* To run the model, run GraphFlow.ipynb in colab or follow below steps

Install prerequisites<br>
	```
	pip install torch==0.4.1.post2 -f https://download.pytorch.org/whl/torch_stable.html 
	```
	<br>
	```
	pip install -r requirements.txt
	```
	<br><br>
Make changes in config/graphflow_dynamic_graph_coqa.yml to pass model parameters.<br><br>
Run below command to train the model <br>
	```
	python main.py -config config/graphflow_dynamic_graph_coqa.yml
	```
	<br>
Logs(Metrics-F1 score and EM) will be saved in data/


## Reference
Yu Chen, Lingfei Wu, Mohammed J. Zaki. **"Graphflow: Exploiting Conversation Flow with Graph Neural Networks for Conversational Machine Comprehension."** In *Proceedings of the 29th International Joint Conference on Artificial Intelligence (IJCAI 2020)*, Yokohama, Japan, Jul 11-17, 2020.


    @article{chen2019graphflow,
      title={Graphflow: Exploiting conversation flow with graph neural networks for conversational machine comprehension},
      author={Chen, Yu and Wu, Lingfei and Zaki, Mohammed J},
      journal={arXiv preprint arXiv:1908.00059},
      year={2019}
    }
