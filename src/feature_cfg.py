# ====================================================
# CFG
# ====================================================
class FE:
    all_features = ['userID', 'Timestamp', 'assessmentItemID', 'testId', 
       'answerCode', 'answerCode_copy', 'KnowledgeTag',
       'test_answer_shift', 'user_correct_answer', 'user_total_answer',
       'user_cum_prob', 'down', 'user_Item_total_answer',
       'user_Item_correct_answer', 'user_Item_cum_prob', 'assessment_main',
       'assessment_mid', 'assessment_sub', 'problem_time',
       'assessment_main_problem_mean', 'assessment_main_problem_count',
       'assessment_mid_problem_mean', 'assessment_mid_problem_count',
       'assessment_sub_problem_mean', 'assessment_sub_problem_count',
       'test_prob', 'test_sum', 'tag_prob', 'tag_sum', 'Item_prob', 'Item_sum',
       'year', 'month', 'day', 'hour', 'minute', 'second', 'weekday',
       'hour_mode', 'MA_prec_all_over5', 'MA_prec_all_over10',
       'MA_prec_all_over20', 'MA_prec_all_over40', 'MA_prec_all_over80',
       'MA_prec_all_over160', 'MA_prec_all_over320', 'MA_prec_by_main5',
       'MA_prec_by_main10', 'MA_prec_by_main50', 'MA_prec_by_main100',
       'MA_prec_by_mid3', 'MA_prec_by_mid6', 'MA_prec_by_mid10',
       'MA_prec_by_sub3', 'MA_prec_by_sub6', 'MA_prec_by_sub10',
       'MA_prec_by_testid3', 'MA_prec_by_testid6']
    
    to_use = ['userID', 'assessmentItemID', 'testId', 
            'answerCode', 'KnowledgeTag', 'assessment_main',
            'assessment_mid', 'assessment_sub', 'problem_time',
            'year', 'month', 'day', 'hour', 'minute', 'second', 'weekday'
    ]

    

    features = to_use


# ['Timestamp', 'answerCode_copy',  
# 				'year', 'day', 'weekday', 'minute', 'second', 'answerCode', 'assessment_mid_mean',
#             'Item_mean', 'assessment_main_count', 'assessment_sub_count', 'assessment_mid_count',
#             'assessment_main_mean', 'assessment_sub_mean', 'Item_mean_cum', 'Item_sum', 'test_mean'
#     ]