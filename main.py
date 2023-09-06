import os, time, logging
from preprocess.parse_process import Parse_Process
from evaluate import Evaluate_Model

def main(args):
    sub_dirs = args.get_sub_dirs()
    for sub_dir in sub_dirs:
        for dir, _, files in os.walk(sub_dir):
            model, model_file_path, true_target_label = args.process_directory(dir, files)
            submodel = args.get_submodel(model).to(args.device)
            if args.phase == 'evaluate': 
                Evaluate_Model(model, submodel, model_file_path, true_target_label, args)
            else: 
                logging.info('Option [{}] is not supported!'.format(args.phase))

            logging.info(f"{'*'*50}\n")

if __name__ == '__main__':
    args = Parse_Process()
    start = time.time(); main(args); end = time.time()
    logging.info(f'Running time: {(end - start) / 60:.4f} m')