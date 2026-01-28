from helpers import reverse_history, Config, write_to_log, convert_string_to_array
from model import Model
from prompts import generate_animals_system_prompt, generate_more_animals_system_prompt, \
    answer_question_yesno_system_prompt, generate_animals_user_prompt


def generate_new_beliefs(system_prompt: dict[str, str], history_questioner: list[dict[str, str]],
                         questioner: Model, generation_temperature: float) -> list[str]:
    messages = ([system_prompt] + reverse_history(history_questioner) + [generate_animals_user_prompt()])
    new_beliefs = questioner.chat_complete(messages=messages, temperature=generation_temperature)[0]
    return convert_string_to_array(new_beliefs)


def check_beliefs_batched(beliefs: list[str], history_questioner: list[dict[str, str]], checker: Model,
                          answer_temperature: float, block_size: int, threshold_rejection_probability: float) -> list[str]:
    if len(beliefs) == 0 or len(history_questioner) == 0:
        return beliefs

    filtered_beliefs = []

    conversations = []
    answers = []
    # generate one conversation for each belief and question to check the belief for
    for new_belief in beliefs:
        for (i, history_message) in enumerate(history_questioner):
            if history_message["role"] != "assistant":
                continue
            question = history_message["content"]
            user_question = {"role": "user", "content": question}
            conversations.append([answer_question_yesno_system_prompt(entity=new_belief), user_question])
            answers.append(history_questioner[i+1]["content"].lower())

    # get answers for all conversations in parallel
    probabilities = checker.chat_probabilities_messages_batched(conversations, ["Yes", "No"],
                                                                temperature=answer_temperature, block_size=block_size)

    # for each belief, check if all question-answer-pairs fit and filter out otherwise
    for i, new_belief in enumerate(beliefs):
        valid_belief = True
        probs_belief = probabilities[i*int(len(history_questioner)/2):(i+1)*int(len(history_questioner)/2)]
        for j, answer in enumerate(probs_belief):
            if ((answer["Yes"] > (1 - threshold_rejection_probability) and answers[j] == "no")
                    or (answer["Yes"] < threshold_rejection_probability and answers[j] == "yes")):
                valid_belief = False
                break
        if valid_belief:
            filtered_beliefs.append(new_belief)

    return filtered_beliefs


def update_beliefs_batched(history: list[(str, str)], beliefs: list[str], questioner: Model, deterministic: bool,
                           config: Config) -> list[str]:
    generation_temperature, max_num_samples, min_num_samples = config.generation_temperature_diverse, config.max_num_samples, config.min_num_samples
    answer_temperature, block_size, threshold_rejection_probability = config.answer_temperature, config.batched_block_size, config.threshold_rejection_probability

    # generate new beliefs
    system_prompt = generate_animals_system_prompt(max_num_samples, min_num_samples)
    beliefs_new = generate_new_beliefs(system_prompt, history, questioner, generation_temperature)

    # in split baseline, return the beliefs sampled using the current history
    if deterministic:
        return beliefs_new

    # filter new beliefs according to previous questions+answers
    beliefs_new = check_beliefs_batched(beliefs_new, history[:-2], questioner, answer_temperature, block_size, threshold_rejection_probability)

    # filter previous beliefs with new question+answer
    filtered_beliefs_old = check_beliefs_batched(beliefs, history[-2:], questioner, answer_temperature, block_size, threshold_rejection_probability)
    # throw out duplicates
    beliefs_updated = list({s.lower(): s for s in beliefs_new + filtered_beliefs_old}.values())

    # try to generate new beliefs twice more if no sufficient number could be generated
    for _ in range(2):
        if len(beliefs_updated) >= min_num_samples:
            break
        system_prompt = generate_more_animals_system_prompt(beliefs_updated, min_num_samples - len(beliefs_updated))
        beliefs_new = generate_new_beliefs(system_prompt, history, questioner, generation_temperature)
        beliefs_new = check_beliefs_batched(beliefs_new, history[:-2], questioner, answer_temperature, block_size, threshold_rejection_probability)
        beliefs_updated = list({s.lower(): s for s in beliefs_new + beliefs_updated}.values())

    # If no valid beliefs at all can be generated, generate unfiltered to continue the game
    if len(beliefs_updated) == 0:
        beliefs_updated = generate_new_beliefs(system_prompt, history, questioner, generation_temperature)

    return beliefs_updated
