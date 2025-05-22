python dataset_mapper_splitted.py --dataset ml100k
cd models/PGPR
python preprocess.py --dataset ml100k
python train_transe_model.py --dataset ml100k
python train_agent.py --dataset ml100k
python test_agent.py --dataset ml100k
cd ../../
mkdir ./logs
python main.py --dataset=ml100k --opt=softETD > logs/ml100k_softETD.log
python main.py --dataset=ml100k --opt=softSEP > logs/ml100k_softSEP.log
python main.py --dataset=ml100k --opt=softLIR > logs/ml100k_softLIR.log
python main.py --dataset=ml100k --opt=ETDopt > logs/ml100k_ETDopt.log
python main.py --dataset=ml100k --opt=SEPopt > logs/ml100k_SEPopt.log
python main.py --dataset=ml100k --opt=LIRopt > logs/ml100k_LIRopt.log
python main.py --dataset=ml100k --opt=ETD_SEP_opt > logs/ml100k_ETD_SEP_opt.log
python main.py --dataset=ml100k --opt=ETD_LIR_opt > logs/ml100k_ETD_LIR_opt.log
python main.py --dataset=ml100k --opt=SEP_LIR_opt > logs/ml100k_SEP_LIR_opt.log
python main.py --dataset=ml100k --opt=ETD_SEP_LIR_opt > logs/ml100k_ETD_SEP_LIR_opt.log