import os
from langchain.evaluation import CriteriaEvalChain
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from src.chatbot import AIChatbot
from dotenv import load_dotenv
import json
from datetime import datetime

load_dotenv()

class ChatbotEvaluator:
    def __init__(self, model_name: str = "llama3-8b-8192"):
        # Use the specified GROQ_MODEL
        GROQ_MODEL = "llama3-8b-8192"
        self.llm = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name=GROQ_MODEL,
            temperature=0.1
        )
        # Initialize chatbot with the same model
        self.chatbot = AIChatbot(model_name=GROQ_MODEL)

        # Initialize evaluation chains
        self.relevance_eval = CriteriaEvalChain.from_llm(
            llm=self.llm,
            criteria="relevance",
            chain_type="stuff"
        )

        self.coherence_eval = CriteriaEvalChain.from_llm(
            llm=self.llm,
            criteria="coherence",
            chain_type="stuff"
        )

        self.groundedness_eval = CriteriaEvalChain.from_llm(
            llm=self.llm,
            criteria="groundedness",
            chain_type="stuff"
        )

    def evaluate_response(self, question, answer, context=None, chat_history=None):
        """Evaluate a single response with comprehensive metrics"""
        evaluations = {}

        try:
            # Relevance evaluation
            relevance_result = self.relevance_eval.evaluate_strings(
                prediction=answer,
                input=question
            )
            evaluations['relevance'] = {
                'score': relevance_result['score'],
                'reasoning': relevance_result['reasoning']
            }

            # Coherence evaluation
            coherence_result = self.coherence_eval.evaluate_strings(
                prediction=answer,
                input=question
            )
            evaluations['coherence'] = {
                'score': coherence_result['score'],
                'reasoning': coherence_result['reasoning']
            }

            # Groundedness evaluation (if context provided)
            if context:
                groundedness_result = self.groundedness_eval.evaluate_strings(
                    prediction=answer,
                    reference=context
                )
                evaluations['groundedness'] = {
                    'score': groundedness_result['score'],
                    'reasoning': groundedness_result['reasoning']
                }

            # Helpfulness evaluation
            helpfulness_result = self.helpfulness_eval.evaluate_strings(
                prediction=answer,
                input=question
            )
            evaluations['helpfulness'] = {
                'score': helpfulness_result['score'],
                'reasoning': helpfulness_result['reasoning']
            }

            # Creativity evaluation
            creativity_result = self.creativity_eval.evaluate_strings(
                prediction=answer,
                input=question
            )
            evaluations['creativity'] = {
                'score': creativity_result['score'],
                'reasoning': creativity_result['reasoning']
            }

            # Completeness evaluation
            completeness_result = self.completeness_eval.evaluate_strings(
                prediction=answer,
                input=question
            )
            evaluations['completeness'] = {
                'score': completeness_result['score'],
                'reasoning': completeness_result['reasoning']
            }

            # Calculate overall score
            scores = [v['score'] for v in evaluations.values() if isinstance(v, dict) and 'score' in v]
            evaluations['overall_score'] = sum(scores) / len(scores) if scores else 0

        except Exception as e:
            evaluations['error'] = str(e)

        return evaluations

    def evaluate_test_set(self, test_questions):
        """Evaluate a set of test questions"""
        results = []

        for question in test_questions:
            print(f"Evaluating: {question}")

            # Get chatbot response
            response = self.chatbot.chat(question)
            answer = response.get("answer", "")
            sources = response.get("sources", [])

            # Extract context from sources
            context = "\n".join([doc.page_content for doc in sources]) if sources else ""

            # Evaluate response
            evaluation = self.evaluate_response(question, answer, context)

            result = {
                "question": question,
                "answer": answer,
                "context": context,
                "evaluation": evaluation
            }

            results.append(result)

        return results

    def generate_report(self, results, output_file="evaluation_report.json"):
        """Generate comprehensive evaluation report"""
        # Calculate averages for all metrics
        metrics = ['relevance', 'coherence', 'groundedness', 'helpfulness', 'creativity', 'completeness', 'overall_score']
        totals = {metric: 0 for metric in metrics}
        count = 0

        for result in results:
            eval_data = result['evaluation']
            for metric in metrics:
                if metric in eval_data:
                    totals[metric] += eval_data[metric]['score']
                    if metric != 'overall_score':  # Count once per result
                        count += 1

        # Calculate averages
        averages = {}
        for metric in metrics:
            if metric == 'overall_score':
                averages[f"average_{metric}"] = totals[metric] / len(results) if results else 0
            else:
                averages[f"average_{metric}"] = totals[metric] / count if count > 0 else 0

        # Calculate performance insights
        insights = self._generate_performance_insights(results, averages)

        report = {
            "summary": {
                "total_questions": len(results),
                **averages
            },
            "performance_insights": insights,
            "detailed_results": results,
            "metadata": {
                "evaluation_timestamp": json.dumps({"timestamp": str(datetime.now())}),
                "model_used": "llama3-8b-8192",
                "evaluation_version": "2.0"
            }
        }

        # Save to file
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)

        return report

    def _generate_performance_insights(self, results, averages):
        """Generate performance insights based on evaluation results"""
        insights = {
            "strengths": [],
            "weaknesses": [],
            "recommendations": []
        }

        # Analyze strengths
        if averages.get('average_relevance', 0) > 0.8:
            insights["strengths"].append("Excellent relevance in responses")
        if averages.get('average_coherence', 0) > 0.8:
            insights["strengths"].append("High coherence in answer structure")
        if averages.get('average_groundedness', 0) > 0.8:
            insights["strengths"].append("Strong grounding in provided context")
        if averages.get('average_helpfulness', 0) > 0.8:
            insights["strengths"].append("Highly helpful responses")
        if averages.get('average_creativity', 0) > 0.7:
            insights["strengths"].append("Creative and engaging answers")

        # Analyze weaknesses
        if averages.get('average_relevance', 0) < 0.6:
            insights["weaknesses"].append("Responses sometimes off-topic")
        if averages.get('average_coherence', 0) < 0.6:
            insights["weaknesses"].append("Answer structure needs improvement")
        if averages.get('average_groundedness', 0) < 0.6:
            insights["weaknesses"].append("Better context utilization needed")
        if averages.get('average_completeness', 0) < 0.6:
            insights["weaknesses"].append("Responses lack completeness")

        # Generate recommendations
        if averages.get('average_overall_score', 0) > 0.8:
            insights["recommendations"].append("Excellent performance - maintain current approach")
        elif averages.get('average_overall_score', 0) > 0.6:
            insights["recommendations"].append("Good performance - focus on fine-tuning specific metrics")
        else:
            insights["recommendations"].append("Needs improvement - review prompt engineering and RAG setup")

        insights["recommendations"].append("Consider implementing query expansion for better retrieval")
        insights["recommendations"].append("Add response caching for frequently asked questions")

        return insights

def run_evaluation():
    """Run evaluation on sample questions"""
    evaluator = ChatbotEvaluator()

    # Sample test questions (you should replace these with your actual test questions)
    test_questions = [
        "What is the main topic of this document?",
        "Can you explain the key concepts discussed?",
        "What are the main benefits mentioned?",
        "How does this system work?",
        "What are the requirements for implementation?"
    ]

    print("Starting evaluation...")
    results = evaluator.evaluate_test_set(test_questions)

    print("Generating report...")
    report = evaluator.generate_report(results)

    print("Evaluation completed!")
    print(f"Average Relevance: {report['summary']['average_relevance']:.2f}")
    print(f"Average Coherence: {report['summary']['average_coherence']:.2f}")
    print(f"Average Groundedness: {report['summary']['average_groundedness']:.2f}")

    return report

if __name__ == "__main__":
    run_evaluation()
