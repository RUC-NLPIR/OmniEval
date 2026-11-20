# -*- coding: utf-8 -*-
import sys
import loguru

sys.path.append('.')
from utils.llm_models import MyLLM
from utils.evaluation import f1_score, exact_match_score, rouge_score, normalize_answer, bert_score
import json
from utils.utils import convert_to_float
from utils.corpus_loader import Corpus
from llama_index.core.schema import MetadataMode
from prompts import relevance_prompt_map, generation_prompt_map, rejection_output
import json_repair
import sys
from utils.options import get_options
import os
from typing import List
import numpy as np
from tqdm import tqdm
from collections import defaultdict

class Judger:
    def __init__(self, opt):
        self.opt = opt
        if self.opt.judge_type == 'model':
            self.gpt_judger = MyLLM(opt)
        
        self.corpus = Corpus(opt) #opt.node_root
        self.corpus.load_corpus_from_nodes(thread=self.opt.thread)
        loguru.logger.info(f'=====> For Judger self.corpus node cnt = {len(self.corpus.nodes)}')
        self.doc_format_str = 'Title: {title}\nContent: {content}'

        self.skip_tasks = ['结构化知识问答']
        self.skip_topics = ['金融科技 - 移动支付', '保险 - 责任保险', '监管与合规 - 客户知识产权（KYC）']

    def clean_data_format(self, pred_results):
        for i in range(len(pred_results)):
            if isinstance(pred_results[i]['relevant_node'], str):
                pred_results[i]['relevant_node'] = [pred_results[i]['relevant_node']]
            if isinstance(pred_results[i]['relevant_passage'], str):
                pred_results[i]['relevant_passage'] = [pred_results[i]['relevant_passage']]
            if isinstance(pred_results[i]['golden_answers'], str):
                pred_results[i]['golden_answers'] = [pred_results[i]['golden_answers']]
        return pred_results

    def load_eval_data(self, pred_results_: str):
        if isinstance(pred_results_, str):
            loguru.logger.info(f'=======> Evaluate from path: {pred_results_}')
            pred_results = [json.loads(line) for line in open(pred_results_, encoding='utf8').readlines()]
        else:
            pred_results = [convert_to_float(data) for data in pred_results_]
        pred_results = self.clean_data_format(pred_results)

        return pred_results
    
    def _MAP(self, rel_node: list, retrieval_results: list, k: int):
        rel_ids = [node.id_  for node in rel_node]
        rel_sum = 0.0
        rel_cnt = 0
        for i in range(min(len(retrieval_results), k)):
            if retrieval_results[i]['id'] in rel_ids:
                rel_sum += (rel_cnt+1) / (i+1)
                rel_cnt += 1

        return rel_sum / len(set(rel_ids)) if rel_ids else 0.0
    
    def _MRR(self, rel_node: list, retrieval_results: list, k: int):
        rel_ids = [node.id_  for node in rel_node]
        metric = 0.0
        for i in range(min(len(retrieval_results), k)):
            if retrieval_results[i]['id'] in rel_ids:
                metric += 1 / (i+1)
                break
        return metric
    
    def _auto_matching_metric(self, question: str, answers: List[str], relevant_str: str, relevant_passage: str, retrieval_result: dict, metric: str ='relevance'):
        if metric not in relevance_prompt_map:
            raise ValueError(f"Not support auto-retrieval metric: {metric}")
        
        system_prompt = relevance_prompt_map[metric][0]
        user_prompt = relevance_prompt_map[metric][1]
       
        title = retrieval_result['title']
        contents = retrieval_result['contents']
        retrieve_str = self.doc_format_str.format(title=title, content=contents)

        user_input = user_prompt.format(question=question, answers='\n'.join(answers), rel_str=relevant_str, rel_psg=relevant_passage, retrieve_str=retrieve_str)

        flag = False
        
        eval_res = json_repair.loads(self.gpt_judger.get_llm_output([system_prompt, user_input], use_batch=False)[0])['prediction']
        
        try:
            eval_res = int(eval_res) / 2 
            flag = True
        except:
            flag = False 

        if not flag:
            eval_res = -1
        return eval_res
    
    def _auto_ranking_metric(self, question: str, answers: List[str], relevant_passage: str, relevant_node: list, retrieval_results: list, k: int, metric: str = 'relevance'):
        format_str = self.doc_format_str 
        relevant_title = [node.metadata['Title'] for node in relevant_node]
        relevant_content = [node.get_content(MetadataMode.NONE) for node in relevant_node]
        
        relevant_str = '\n'.join([format_str.format(title=relevant_title[i], content=relevant_content[i]) for i in range(len(relevant_title))])
        rel_sum = 0.0
        for i, retrieval_result in enumerate(retrieval_results):
            if i >= k: break
            eval_res = self._auto_matching_metric(question, answers, relevant_str, relevant_passage, retrieval_result, metric=metric)
            if eval_res >= 0:
                rel_sum += eval_res / (i+1)
            
        return rel_sum / min(k, len(retrieval_results))

    def retrieval_eval(self, pred_results: List[dict]):
        rule = {
            "MRR": [],
            "MAP": [],
        }
        gpt_eval = {
            "relevance": [],
            "necessity": [],
        }
        if self.opt.judge_type == 'all':
            result_map = dict(list(rule.item()) + list(gpt_eval.item()))
        elif self.opt.judge_type == 'rule':
            result_map = rule
        else:
            result_map = gpt_eval

        rule_map = {
            "MRR": self._MRR,
            "MAP": self._MAP, 
        }
        for data in tqdm(pred_results, desc='Evaluation for retrieval metrics..'):
            
            question = data['question']
            answers = data['golden_answers']
            output = data['output']
            retrieval_results = output['retrieval_result']
            relevant_passage = '\n'.join(data['relevant_passage'])
            try:
                relevant_nodes = [self.corpus.node_map[node_id] for node_id in data['relevant_node']]
            except:
                assert False, f'=====> data = {data} Cannot find relevant_nodes'

            k = len(retrieval_results)

            for metric in result_map:
                if metric in rule_map:
                    res = rule_map[metric](relevant_nodes, retrieval_results, k)
                else:
                    res = self._auto_ranking_metric(question, answers, relevant_passage, relevant_nodes, retrieval_results, k, metric=metric)
                result_map[metric].append(res)

        for key in result_map:
            loguru.logger.info(f"key: {key}, result_map[key]: {result_map[key]}")
            res_list = result_map[key]
            result_map[key] = {
                'mean': float(self.get_valid_mean(res_list)) if (isinstance(res_list[0], float) or isinstance(res_list[0], int)) else  np.mean(np.array(res_list), axis=0).tolist(),
                'list': ','.join([str(float(v)) if not isinstance(v, tuple) else '-'.join([str(v_) for v_ in v]) for v in res_list])
            }

        return result_map

    def _auto_generation_metric(self, question: str | List[str], answers: List[str] | List[List[str]], pred: str | List[str], doc_str: str | List[str], metric: str | List[str]):
        if not isinstance(question, str):
            return self._auto_generation_metric_batch(question, answers, pred, doc_str, metric)

        if metric not in generation_prompt_map:
            raise ValueError(f"Not support auto-generation metric: {metric}")
        
        system_prompt = generation_prompt_map[metric][0]
        user_prompt = generation_prompt_map[metric][1] 
    
        user_input = user_prompt.format(question=question, answers='\n'.join(answers), prediction=pred, doc_str=doc_str) 
        
        try:
            eval_res = json_repair.loads(self.gpt_judger.get_llm_output([system_prompt, user_input], use_batch=False)[0])
        except:
            eval_res = ''

        if not isinstance(eval_res, dict):
            eval_res = {'prediction': -1}

        if 'prediction' in eval_res:
            eval_res = eval_res['prediction'] 
        elif metric in eval_res:
            eval_res = eval_res[metric]  
        
        if metric not in ['numerical_accuracy', 'hallucination'] :
            eval_res = int(eval_res) / 2 
        else:
            eval_res = int(eval_res)

        return eval_res
    
    def _auto_generation_metric_batch(self, question: List[str], answers: List[List[str]], pred: List[str], doc_str: List[str], metric: List[str]): 
        for m in metric:
            if m not in generation_prompt_map:
                raise ValueError(f"Not support auto-generation metric: {metric}")

        bsz = len(question)    
        all_inputs = []
        for i in range(bsz):
            system_prompt = generation_prompt_map[metric[i]][0]
            user_prompt = generation_prompt_map[metric[i]][1] 
            user_input = user_prompt.format(question=question[i], answers='\n'.join(answers[i]), prediction=pred[i], doc_str=doc_str[i]) 
            all_inputs.append([system_prompt, user_input])

        all_eval_data_list = self.gpt_judger.get_llm_output(all_inputs, use_batch=True)
        all_eval_res = []
        for i, eval_data in enumerate(all_eval_data_list):
            try:
                eval_res = json_repair.loads(eval_data)
            except:
                eval_res = ''

            if not isinstance(eval_res, dict) or len(eval_res) == 0:
                eval_res = {'prediction': -1}

            loguru.logger.info(f'=====> eval_res=  {eval_res} type = {type(eval_res)}')
            m = metric[i]
            if 'prediction' in eval_res:
                eval_res = eval_res['prediction'] 
            elif m in eval_res:
                eval_res = eval_res[m] 
            else:
                eval_res = -1
            
            try:
                eval_res = int(eval_res)
            except:
                eval_res = -1
        
            if m not in ['numerical_accuracy', 'hallucination'] and eval_res != -1:
                eval_res = eval_res / 2  

            all_eval_res.append(eval_res)
        return all_eval_res
    
    def get_valid_mean(self, list_):
        if isinstance(list_[0], list) :
            return np.mean(np.array(list_), axis=0)

        valid_list = [item for item in list_ if item >= 0]
        if len(valid_list) == 0: return np.array(-1)
        return np.mean(np.array(valid_list), axis=0)

    def generation_eval_batch(self, pred_results: List[dict], rejection_results: List[dict] = None, batch_size=1):
        rule = {
            "em": [],
            "rouge": [],
            "f1": [],
        }
        gpt_eval = {metric:[] for metric in self.opt.model_eval_metrics.split(',')}
        loguru.logger.info(f"gpt_eval metrics = {gpt_eval.keys()}")
  
        if self.opt.close_book:
            loguru.logger.info(f"=====> self.opt.close_book = {self.opt.close_book} before remove hallucination and utilization gpt_eval = {gpt_eval}")
            if 'hallucination' in gpt_eval:
                del gpt_eval['hallucination']
            if 'utilization' in gpt_eval:
                del gpt_eval['utilization']
            loguru.logger.info(f"=====> self.opt.close_book = {self.opt.close_book} after remove hallucination and utilization gpt_eval = {gpt_eval}")
        
        if self.opt.judge_type == 'all':
            result_map = dict(list(rule.item()) + list(gpt_eval.item()))
        elif self.opt.judge_type == 'rule':
            result_map = rule
        else:
            result_map = gpt_eval
        rule_map = {
            "em": exact_match_score,
            "rouge": rouge_score,
            "f1": f1_score,
            "bert_score": bert_score,
        }
        
        for bg in tqdm(range(0, len(pred_results), batch_size), desc='Evaluation for generation metrics..'):
            ed = min(len(pred_results), bg + batch_size)
            all_question = []
            all_answers = []
            all_pred = []
            all_metric = []
            all_doc_str = []
            for i in range(bg, ed):
                data = pred_results[i]
                question = data['question']
                answers = data['golden_answers']
                output = data['output']
                pred = output['pred']

                if 'retrieval_result' in output:
                    retrieval_results = output['retrieval_result']
                    format_str = self.doc_format_str 
                    retrieval_title = [result['title'] for result in retrieval_results]
                    retrieval_content = [result['contents'] for result in retrieval_results]
                
                    retrieval_str = '\n'.join([format_str.format(title=retrieval_title[i], content=retrieval_content[i]) for i in range(len(retrieval_title))])
                else: 
                    retrieval_str = None
                for metric in result_map:
                    if metric in generation_prompt_map:
                        all_question.append(question)
                        all_answers.append(answers)
                        all_pred.append(pred)
                        all_metric.append(metric)
                        all_doc_str.append(retrieval_str)
                    else:
                        per_res = rule_map[metric](pred, answers, normalize_answer)
                        result_map[metric].append(per_res)
                        
            if self.opt.judge_type != 'rule':
                all_eval_results = self._auto_generation_metric_batch(all_question, all_answers, all_pred, metric=all_metric, doc_str=all_doc_str)
                for i in range(len(all_metric)):
                    metric = all_metric[i]
                    result_map[metric].append(all_eval_results[i])


        if rejection_results is not None:
            reject_cnt = 0
            for data in rejection_results:
                output = data['output']
                pred = output['pred']

                if 'rejection_output' in pred:
                    reject_cnt += 1
            result_map['rejection'] = reject_cnt / len(rejection_results)

        for key in result_map:
            res_list = result_map[key]
            result_map[key] = {
                'mean': float(self.get_valid_mean(res_list)) if (isinstance(res_list[0], float) or isinstance(res_list[0], int)) else  np.mean(np.array(res_list), axis=0).tolist(),
                'list': ','.join([str(float(v)) if not isinstance(v, tuple) else '-'.join([str(v_) for v_ in v]) for v in res_list])
            }

        return result_map

    def _evaluation(self, pred_results_: str, rejection_results_: str = None, eval_retrieve: bool = True, eval_generation: bool = True):
        pred_results = self.load_eval_data(pred_results_)
        if rejection_results_ is not None:
            rejection_results = self.load_eval_data(rejection_results_)
        else:
            rejection_results = None
        if eval_retrieve:
            retrieval_res_map = self.retrieval_eval(pred_results)
        else:
            retrieval_res_map = None
        if eval_generation:
            generation_res_map = self.generation_eval_batch(pred_results, rejection_results, opt.eval_batch_size)
        else:
            generation_res_map = None

        return retrieval_res_map, generation_res_map

    def evaluation(self):
        data_root_name = self.opt.data_gen_root.split('/')[-1]
        
        if self.opt.eval_suffix == 'none':
            self.opt.eval_suffix = ""
            
        self.opt.generator_model = self.opt.generator_model.split(',')
        self.opt.retrieval_method = self.opt.retrieval_method.split(',')
        
        for generator_model in self.opt.generator_model:
            if self.opt.close_book:
                self.opt.retrieval_method = ['CLOSE']
                eval_retrieve = False
            else:
                eval_retrieve = self.opt.judge_type == 'rule'
           
            for retrieval_method in self.opt.retrieval_method :
                if retrieval_method != 'CLOSE':
                    model_name = retrieval_method + f'_TOP{self.opt.retrieval_topk}' + '-' + generator_model
                else:
                    model_name = retrieval_method + '-' + generator_model

                res_root = os.path.join(self.opt.pred_result_root, data_root_name, model_name)
                if not os.path.exists(res_root):
                    loguru.logger.info(f'=====> Dont exists {res_root} skip to evaluate this model....')
                    continue
                    
                if opt.eval_suffix != "":
                    if opt.eval_suffix[0] != '_':
                        eval_suffix = '_' + opt.eval_suffix
                    else:
                        eval_suffix = opt.eval_suffix
                else:
                    eval_suffix = ""
                save_path = os.path.join(res_root, f'evaluation_result_{self.opt.judge_type}{eval_suffix}.jsonl')

                have_res_dict = {}
                if os.path.exists(save_path):
                    have_data = [json.loads(line) for line in open(save_path, encoding='utf-8-sig').readlines()]
                    for data in have_data:
                        have_res_dict.update(data)

                for file in os.listdir(res_root):
                    if '.jsonl' not in file or 'evaluation_result' in file: continue 
                    if file in have_res_dict: 
                        loguru.logger.info(f'======> Already evaluated {file}, skip...')
                        continue
                    items = file.split('.pred.jsonl')[0].split(' - ')
                    topic = ' - '.join(items[:-1])
                    task = items[-1]
                    if task in self.skip_tasks: 
                        continue 
                    if topic in self.skip_topics:
                        continue

                    path = os.path.join(res_root, file)
                    loguru.logger.info(f'======> Begin to evaluate {file}')
                    retrieval_res_map, generation_res_map = self._evaluation(path, eval_retrieve=eval_retrieve)
                    loguru.logger.info(f'======> Evaluate {file} Over')
                    add_data = {
                        file: {
                            "retrieval_res_map": retrieval_res_map, 
                            "generation_res_map": generation_res_map,
                        }
                    }
                    with open(save_path, 'a+', encoding='utf8') as wf:
                        wf.write(json.dumps(add_data, ensure_ascii=False) + '\n')

                if os.path.exists(save_path):
                    raw_result_ = [json.loads(line) for line in open(save_path, encoding='utf8').readlines()]
                    raw_result = []
                    for data in raw_result_:
                        keys = list(data.keys())
                        flag = True
                        if len(keys) == 1 and '.pred.jsonl' in keys[0]:
                            fn = keys[0]
                            items = fn.split('.pred.jsonl')[0].split(' - ')
                            topic = (' - ').join(items[:-1])
                            task = items[-1]
                            if topic in self.skip_topics or task in self.skip_tasks:
                                flag = False
                        else:
                            flag = False
                        if flag:
                            raw_result.append(data)

                    # calculate all average for all subset
                    sort_result = sorted(raw_result, key=lambda x: list(x.keys())[0])
                    
                    # all average:
                    all_ret_res = defaultdict(list)
                    all_gen_res = defaultdict(list)
                    valid_ret = True
                    valid_gen = True
                    for result_ in sort_result:
                        result = list(result_.values())[0]
                        assert result is not None, f"result is None, result_ = {result_}"
                        if 'retrieval_res_map' not in result or result['retrieval_res_map'] is None:
                            valid_ret = False
                        else:
                            ret_res = result['retrieval_res_map']
                            for subkey in ret_res:
                                all_ret_res[subkey] += [float(v) if '-' not in v else [float(v_) for v_ in v.split('-')] for v in ret_res[subkey]['list'].split(',')]
                        if 'generation_res_map' not in result:
                            valid_gen = False
                        else:
                            gen_res = result['generation_res_map']
                            for subkey in gen_res:
                                new_res = []
                                for v in gen_res[subkey]['list'].split(','):
                                    try:
                                        v = float(v)
                                    except:
                                        v = [float(v_) for v_ in v.split('-')]
                                    new_res.append(v)
                                all_gen_res[subkey] += new_res
                                
                    if valid_ret:
                        for key in all_ret_res:
                            all_ret_res[key] = self.get_valid_mean(all_ret_res[key]) 
                            if len(all_ret_res[key].shape) == 0:
                                all_ret_res[key] = float(all_ret_res[key])
                            else:
                                all_ret_res[key] = ','.join([str(v) for v in all_ret_res[key].tolist()])
                    if valid_gen:
                        for key in all_gen_res:
                            all_gen_res[key] = self.get_valid_mean(all_gen_res[key]) 
                            if len(all_gen_res[key].shape) == 0:
                                all_gen_res[key] = float(all_gen_res[key])
                            else:
                                all_gen_res[key]  = ','.join([str(v) for v in all_gen_res[key].tolist()])
                    all_avg_data = {
                        "all_avg_retrieval_res_map": all_ret_res if valid_ret else None, 
                        "all_avg_generation_res_map": all_gen_res if valid_gen else None,
                    }

                    with open(save_path, 'w', encoding='utf8') as wf:
                        for data in sort_result:
                            file = list(data.keys())[0]
                            result = list(data.values())[0]
                            if 'retrieval_res_map' not in result:
                                valid_ret = False
                            else:
                                ret_res = result['retrieval_res_map']
                                result['retrieval_res_map'] = ret_res
                            if 'generation_res_map' not in result:
                                valid_gen = False
                            else:
                                gen_res = result['generation_res_map']
                                result['generation_res_map'] = gen_res
                            data[file] = result

                            wf.write(json.dumps(data, ensure_ascii=False) + '\n')
                        wf.write(json.dumps(all_avg_data, ensure_ascii=False) + '\n')
                    loguru.logger.info(f'======> Evaluate all files of {res_root} Over!')
if __name__ == '__main__':
    
    opt = get_options()
    
    judger = Judger(opt)
    judger.evaluation()
    