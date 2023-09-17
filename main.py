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

    generate_data, validate_data = {}, {}
    for target in range(args.num_classes):
        generate_data[target] = Evaluate_Data(256, args.dataset, 'train', target).load_and_preprocess_data(dataloader, size_per_class=10)
        validate_data[target] = Evaluate_Data(256, args.dataset, 'train', target).load_and_preprocess_data(dataloader, size_per_class=10)

    for sub_dir in sub_dirs:
        for dir, _, files in os.walk(sub_dir):
            model, model_file_path, true_target_label, attack_spec = args.process_directory(dir, files)
            submodel = args.get_submodel(model).to(args.device)
            if args.phase == 'evaluate': 
                eval_model = Evaluate_Model(model, submodel, model_file_path, args)
                time_start = time.time()  
                for target in range(args.num_classes):
                    # if target == true_target_label: # For Debugging Purpose Only
                    #     eval_model.evaluate_and_log_single_target(target, generate_data, validate_data)  
                    eval_model.evaluate_and_log_single_target(target, generate_data, validate_data)  
                eval_model.logger.print_final_results()
                logging.info(f'Generation Time: {(time.time() - time_start) / 60:.4f} m')
            else: 
                logging.info('Option [{}] is not supported!'.format(args.phase))

            logging.info(f"{'*'*50}\n")
    
    
if __name__ == '__main__':
    args = Parse_Process()
    main(args)

