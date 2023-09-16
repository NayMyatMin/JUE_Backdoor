import sys, os, time, logging
list(map(sys.path.append, ['./architecture/', './preprocess/', './weight_mask/', './SWM/']))
from preprocess.parse_process import Parse_Process
from preprocess.evaluate_data import Evaluate_Data
from weight_mask.weight_masking import weight_masking
from evaluate import Evaluate_Model

from SWM.swm import swm

def main(args):
    sub_dirs = args.get_sub_dirs()
    dataloader = Evaluate_Data(args.batch_size, args.dataset, 'train').load_data()
    for sub_dir in sub_dirs:
        for dir, _, files in os.walk(sub_dir):
            model, model_file_path, true_target_label, attack_spec = args.process_directory(dir, files)
            submodel = args.get_submodel(model).to(args.device)
            if args.phase == 'evaluate': 
                eval_model = Evaluate_Model(model, submodel, model_file_path, args, dataloader)
                eval_model.evaluate_all_targets()
            else: 
                logging.info('Option [{}] is not supported!'.format(args.phase))

            logging.info(f"{'*'*50}\n")

            # swm(model, args.dataset, model_file_path, true_target_label, attack_spec)

if __name__ == '__main__':
    args = Parse_Process()
    start = time.time(); main(args); end = time.time()
    logging.info(f'Running time: {(end - start) / 60:.4f} m')