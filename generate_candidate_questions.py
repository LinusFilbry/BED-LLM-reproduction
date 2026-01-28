import numpy as np

from helpers import reverse_history, _binary_entropy, write_to_log, convert_string_to_array
from model import Model
from prompts import candidate_generation_system_message, conditional_question_generation_prompt, \
    unconditional_question_generation_prompt, candidate_generation_system_message_naive, \
    question_generation_prompt_naive, answer_question_yesno_system_prompt


def generate_candidate_questions(beliefs: list[str], history_questioner: list[dict[str, str]],
                                 questioner: Model, generation_temperature: float, num_questions: int) -> list[str]:
    # if there are less than 3 beliefs left, best question is always to check one of them
    if len(beliefs) in [1, 2]:
        return [f"Is it {beliefs[0]}?"]

    messages = ([candidate_generation_system_message()] + reverse_history(history_questioner) +
                [conditional_question_generation_prompt(beliefs, num_questions)])
    candidate_questions = questioner.chat_complete(messages=messages, temperature=generation_temperature)[0]
    candidate_questions = convert_string_to_array(candidate_questions)

    if len(candidate_questions) < num_questions:
        messages = ([candidate_generation_system_message()] + reverse_history(history_questioner) +
                    [unconditional_question_generation_prompt(candidate_questions, num_questions - len(candidate_questions))])
        new_candidate_questions = questioner.chat_complete(messages=messages, temperature=generation_temperature)[0]
        candidate_questions = candidate_questions + convert_string_to_array(new_candidate_questions)

    return candidate_questions


def evaluate_questions_batched(beliefs: list[str], cand_questions: list[str], eig: bool, deterministic: bool,
                                   questioner: Model, answer_temperature: float, num_mc_samples: int, block_size: int) -> list[float]:
    question_value = [0.0]*len(cand_questions)
    if not deterministic:
        # draw according to p_f, i.e. uniformly
        NUM_MC_SAMPLES = num_mc_samples
        if len(beliefs) < NUM_MC_SAMPLES:
            NUM_MC_SAMPLES = len(beliefs)
            samples = beliefs
        else:
            samples = np.random.choice(beliefs, size=NUM_MC_SAMPLES, replace=True)
    else:
        # in split, evaluate on entire current beliefs
        NUM_MC_SAMPLES = len(beliefs)
        samples = beliefs

    conversations = []
    for question in cand_questions:
        for sample in samples:
            user_question = {"role": "user", "content": f"{question}"}
            messages = [answer_question_yesno_system_prompt(entity=sample), user_question]
            conversations.append(messages)

    probabilities = questioner.chat_probabilities_messages_batched(conversations, ["Yes", "No"],
                                                                   temperature=answer_temperature, block_size=block_size)


    for i, question in enumerate(cand_questions):
        answers = probabilities[i*NUM_MC_SAMPLES:(i+1)*NUM_MC_SAMPLES]
        p_yes = []
        p_no =  []
        entropy_sum = []

        for (j, answer) in enumerate(answers):
            p_yes.append(answer["Yes"])
            p_no.append(answer["No"])
            entropy_sum.append(_binary_entropy(answer["Yes"], answer["No"]))

        p_hat_yes = float(np.mean(p_yes))
        p_hat_no = float(np.mean(p_no))
        entropy = _binary_entropy(p_hat_yes, p_hat_no)
        if eig:
            question_value[i] = entropy - float(np.mean(entropy_sum))
        else:
            question_value[i] = entropy

    return question_value


def generate_candidate_question_naive(history_questioner: list[dict[str, str]], questioner: Model,
                                      generation_temperature: float) -> str:
    messages = ([candidate_generation_system_message_naive()] + reverse_history(history_questioner) +
                [question_generation_prompt_naive()])
    return questioner.chat_complete(messages=messages, temperature=generation_temperature)[0]
