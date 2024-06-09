import tkinter as tk
from transformers import pipeline

# Przygotowanie modelu QA z Transformers
qa_pipeline = pipeline("question-answering")

# Przykładowy tekst, na podstawie którego chatbot będzie odpowiadać
context = """
Gabriela worked for a multinational company as a successful project manager in Brazil and was transferred to manage a team in Sweden. She was excited about her new role but soon realised that managing her new team would be a challenge.

Despite their friendliness, Gabriela didn't feel respected as a leader. Her new staff would question her proposals openly in meetings, and when she gave them instructions on how to carry out a task, they would often go about it in their own way without checking with her. When she announced her decisions on the project, they would continue giving their opinions as if it was still up for discussion.

After weeks of frustration, Gabriela emailed her Swedish manager about the issues she was facing with her team. Her manager simply asked her if she felt her team was still performing, and what she thought would help her better collaborate with her team members. Gabriela found her manager vague and didn't feel as if he was managing the situation satisfactorily.

What Gabriela was experiencing was a cultural clash in expectations. She was used to a more hierarchical framework where the team leader and manager took control and gave specific instructions on how things were to be done. This more directive management style worked well for her and her team in Brazil but did not transfer well to her new team in Sweden, who were more used to a flatter hierarchy where decision making was more democratic. When Gabriela took the issue to her Swedish manager, rather than stepping in with directions about what to do, her manager took on the role of coach and focused on getting her to come up with her own solutions instead.

Dutch social psychologist Geert Hofstede uses the concept of 'power distance' to describe how power is distributed and how hierarchy is perceived in different cultures. In her previous work environment, Gabriela was used to a high power distance culture where power and authority are respected and everyone has their rightful place. In such a culture, leaders make the big decisions and are not often challenged. Her Swedish team, however, were used to working in a low power distance culture where subordinates often work together with their bosses to find solutions and make decisions. Here, leaders act as coaches or mentors who encourage independent thought and expect to be challenged.

When Gabriela became aware of the cultural differences between her and her team, she took the initiative to have an open conversation with them about their feelings about her leadership. Pleased to be asked for their thoughts, Gabriela's team openly expressed that they were not used to being told what to do. They enjoyed having more room for initiative and creative freedom. When she told her team exactly what she needed them to do, they felt that she didn't trust them to do their job well. They realised that Gabriela was taking it personally when they tried to challenge or make changes to her decisions, and were able to explain that it was how they'd always worked.

With a better understanding of the underlying reasons behind each other's behaviour, Gabriela and her team were able to adapt their way of working. Gabriela was then able to make adjustments to her management style so as to better fit the expectations of her team and more effectively motivate her team to achieve their goals.
"""

def get_answer(question):
    result = qa_pipeline(question=question, context=context)
    return result['answer']

def ask_question():
    question = entry.get()
    answer = get_answer(question)
    chat_log.config(state=tk.NORMAL)
    chat_log.insert(tk.END, "You: " + question + "\n")
    chat_log.insert(tk.END, "Bot: " + answer + "\n\n")
    chat_log.config(state=tk.DISABLED)
    entry.delete(0, tk.END)

# Konfiguracja interfejsu użytkownika
root = tk.Tk()
root.title("Chatbot QA")

chat_log = tk.Text(root, state=tk.DISABLED, width=80, height=20)
chat_log.pack()

entry = tk.Entry(root, width=80)
entry.pack(pady=10)

ask_button = tk.Button(root, text="Ask", command=ask_question)
ask_button.pack()

root.mainloop()
