from Scorer import Scorer
import time
import json

if __name__ == '__main__':
    #初始化模型
    t = time.time()
    scorer = Scorer()
    print(f'加载模型耗时：{time.time()-t}s')
    print()

    #读入测试数据
    test1 = {'text': '哦，我这边是伴鱼绘本的课程老师。我看我这边我。试听课在伴鱼绘本对吧。没有解释，就是这么规定的',
            'standard': ['宝妈你好，我是伴鱼少儿英语的老师。', '您现在方便吗？'],
            'keywords': ['伴鱼', '少儿英语'],
            }
    test2 = {'text': '诶你好宝妈，我是伴鱼少儿英语的老师。有什么需要帮助吗',
            'standard': ['宝妈你好，我是伴鱼少儿英语的老师。', '您现在方便吗？'],
            'keywords': ['伴鱼', '少儿英语'],
            }
    test_texts = [test1, test2]

    #循环打分
    results = []
    for test_text in test_texts:
        start = time.time()
        result = {}
        text = test_text['text']
        standard = test_text['standard']
        keywords = test_text['keywords']

        t = time.time()
        fluency_score, fluency_result = scorer.fluency(text)
        result['fluency_result'] = fluency_result
        result['fluency_score'] = fluency_score
        print(f'流利度打分耗时：{time.time() - t}s')

        t = time.time()
        keywords_score, keywords_result = scorer.keywords(text, keywords)
        result['keywords_result'] = keywords_result
        result['keywords_score'] = keywords_score
        print(f'关键词打分耗时：{time.time() - t}s')

        t = time.time()
        pp_score, pp_result = scorer.politeness_prohibition(text)
        result['pp_result'] = pp_result
        result['pp_score'] = pp_score
        print(f'文明用语与服务禁语打分耗时：{time.time() - t}s')

        # t = time.time()
        # similarity_score, similarity_result = scorer.sent_sim(text, standard)
        # result['similarity_result'] = similarity_result
        # result['similarity_score'] = similarity_score
        # print(f'表述准确打分耗时：{time.time() - t}s')

        t = time.time()
        content_score, content_result, similarity_score, similarity_result = scorer.content_completeness(text, standard)
        result['similarity_result'] = similarity_result
        result['similarity_score'] = similarity_score
        result['content_result'] = content_result
        result['content_score'] = content_score
        print(f'语义相似度和内容完整度打分耗时：{time.time() - t}s')

        results.append(result)
        print(f'总用时：{time.time()-start}')
        print()

    json_str = json.dumps(results, indent=4, ensure_ascii=False)
    with open('results.json', 'w') as json_file:
        json_file.write(json_str)
