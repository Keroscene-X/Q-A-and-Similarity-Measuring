import customtkinter as ctk
from functools import partial
from tkinter.filedialog import askopenfilename
from tkinter.messagebox import showinfo
from PIL import Image
import os
import json
from threading import Thread
import pinecone
import sys
import random
import glob
import pandas as pd
import string

from langchain.document_loaders import PyMuPDFLoader
from langchain.vectorstores import Pinecone
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import DirectoryLoader
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.docstore.document import Document
from langchain.document_loaders import AsyncHtmlLoader
from langchain.document_transformers import Html2TextTransformer

ctk.set_appearance_mode("dark")  # Modes: "System" (standard), "Dark", "Light"
# Themes: "blue" (standard), "green", "dark-blue"
ctk.set_default_color_theme("blue")
os.environ["OPENAI_API_KEY"] = "Please Enter Your Key"
score_threshold = 0.83
ranking_threshold = 2
directory = os.path.join(sys.path[0], "data")
embeddings_model = OpenAIEmbeddings()
pinecone.init(api_key="Please Enter Your Key",
              environment="us-west4-gcp-free")
index_name = "document-similarity"
llm = ChatOpenAI(model_name="gpt-3.5-turbo-16k", temperature=1)
chain = load_qa_chain(llm, chain_type="stuff")
numbers = ("1", "2", "3", "4", "5")
uos_prefix = "https://www.sydney.edu.au/units"
admin = 0
password = "0212"


class ToplevelWindow(ctk.CTkToplevel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.geometry("500x400")
        self.title("Settings")
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        image_path = os.path.join(os.path.dirname(
            os.path.realpath(__file__)), "icons")
        self.threshold_image = ctk.CTkImage(Image.open(
            os.path.join(image_path, "threshold.png")), size=(20, 20))
        self.upload_image = ctk.CTkImage(Image.open(
            os.path.join(image_path, "upload.png")), size=(20, 20))

        self.navigation_frame = ctk.CTkFrame(self, corner_radius=0)
        self.navigation_frame.grid(row=0, column=0, sticky="nsew")
        self.navigation_frame.grid_rowconfigure(4, weight=1)

        self.threshold_button = ctk.CTkButton(self.navigation_frame, corner_radius=0, height=40, border_spacing=10, text="Threshold",
                                              fg_color="transparent", text_color=("gray10", "gray90"), hover_color=("gray70", "gray30"), image=self.threshold_image,
                                              anchor="w", command=partial(self.navigation_button_event, "threshold"))
        self.threshold_button.grid(row=1, column=0, sticky="ew")

        self.manage_button = ctk.CTkButton(self.navigation_frame, corner_radius=0, height=40, border_spacing=10, text="Manage",
                                           fg_color="transparent", text_color=("gray10", "gray90"), hover_color=("gray70", "gray30"), image=self.upload_image, anchor="w", command=partial(self.navigation_button_event, "manage"))
        self.manage_button.grid(row=2, column=0, sticky="ew")

        self.threshold_frame = ctk.CTkFrame(
            self, corner_radius=0, fg_color="transparent")
        self.threshold_frame.grid_columnconfigure(0, weight=1)

        self.score_label = ctk.CTkLabel(
            self.threshold_frame, text="Similarity Score Threshold (%):")
        self.score_label.pack(padx=20, pady=(20, 0))

        self.score_entry = ctk.CTkEntry(
            self.threshold_frame, height=30, placeholder_text="83")
        self.score_entry.insert(0, str(score_threshold*100))
        self.score_entry.pack(padx=20, pady=10)

        self.rank_label = ctk.CTkLabel(
            self.threshold_frame, text="Ranking Number Threshold:")
        self.rank_label.pack(padx=20)

        self.rank_entry = ctk.CTkEntry(
            self.threshold_frame, height=30, placeholder_text="2")
        self.rank_entry.insert(0, ranking_threshold)
        self.rank_entry.pack(padx=20, pady=10)

        self.setting_button_ok = ctk.CTkButton(self.threshold_frame, fg_color="transparent", border_width=2, text_color=(
            "gray10", "#DCE4EE"), text="Enter", command=self.set_threshold)
        self.setting_button_ok.pack(padx=20, pady=(10, 0))

        self.admin_frame = ctk.CTkFrame(
            self, corner_radius=0, fg_color="transparent")
        self.admin_frame.grid_columnconfigure(0, weight=1)

        self.admin_label = ctk.CTkLabel(
            self.admin_frame, text="Please enter the password:", font=ctk.CTkFont(size=20, weight="bold"))
        self.admin_label.pack(padx=20, pady=(80, 0))

        self.admin_entry = ctk.CTkEntry(self.admin_frame, height=30)
        self.admin_entry.pack(padx=20, pady=20)

        self.admin_button_ok = ctk.CTkButton(self.admin_frame, fg_color="transparent", border_width=2, text_color=(
            "gray10", "#DCE4EE"), text="Enter", command=self.verify)
        self.admin_button_ok.pack(padx=20, pady=(10, 0))

        self.upload_frame = ctk.CTkFrame(
            self, corner_radius=0, fg_color="transparent")
        self.upload_frame.grid_columnconfigure(0, weight=1)

        self.delete_button = ctk.CTkButton(
            self.upload_frame, text="Delete All Document Embeddings", width=210, height=35, command=self.delete_all)
        self.delete_button.pack(padx=20, pady=(80, 0))

        self.single_doc_button = ctk.CTkButton(
            self.upload_frame, text="Single Document Upload (URL)", width=210, height=35, command=self.popup_threading)
        self.single_doc_button.pack(pady=30, padx=10)

        self.mul_docs_button = ctk.CTkButton(
            self.upload_frame, text="Multiple Documents Upload (Excel)", width=210, height=35, command=self.open_file_threading)
        self.mul_docs_button.pack(padx=10)

        self.manage_frame = self.admin_frame
        if admin == 1:
            self.manage_frame = self.upload_frame

        self.progressbar = ctk.CTkProgressBar(self.upload_frame)
        self.progressbar.configure(mode="indeterminnate")

        self.select_frame_by_name("threshold")

    def select_frame_by_name(self, name):
        global admin
        self.threshold_button.configure(
            fg_color=("gray75", "gray25") if name == "threshold" else "transparent")
        self.manage_button.configure(
            fg_color=("gray75", "gray25") if name == "manage" else "transparent")

        # show selected frame
        if name == "threshold":
            self.threshold_frame.grid(row=0, column=1, sticky="nsew")
        else:
            self.threshold_frame.grid_forget()
        if name == "manage":
            self.manage_frame.grid(row=0, column=1, sticky="nsew")
        else:
            self.manage_frame.grid_forget()

    def navigation_button_event(self, name):
        self.select_frame_by_name(name)

    def verify(self):
        global admin
        if self.admin_entry.get() == password:
            admin = 1
            self.admin_frame.grid_forget()
            self.manage_frame = self.upload_frame
            self.select_frame_by_name("manage")
        else:
            showinfo(title="Error",
                     message="Invalid Password", parent=self)

    def popup_threading(self):

        thread = Thread(target=self.popup,
                        args=(), daemon=True)
        thread.start()

    def popup(self):
        USER_INP = ctk.CTkInputDialog(
            text="Please type in the URL:", title="Single Document Upload (URL)")
        unit_url = USER_INP.get_input()
        if unit_url == None or unit_url == "":
            return
        self.progressbar.pack(padx=20, pady=(10, 0))
        self.progressbar.start()
        try:
            loader = AsyncHtmlLoader([unit_url])
            docs = loader.load()
        except:
            showinfo(title="Warning",
                     message="Invalid URL!", parent=self)
            self.progressbar.pack_forget()
            return
        html2text = Html2TextTransformer()
        docs_transformed = html2text.transform_documents(docs)
        for doc in docs_transformed:
            code = doc.metadata["source"].split("/")[4]
            file_path = os.path.join(directory, code + ".txt")
            f = open(file_path, "w", encoding='utf-8')
            f.write(doc.page_content)
            f.close()
        loader = DirectoryLoader(directory)
        documents = loader.load()
        Pinecone.from_documents(
            documents, embeddings_model, index_name=index_name)
        file_paths = glob.glob(os.path.join(directory, "*"))
        for file_path in file_paths:
            os.remove(file_path)
        self.progressbar.pack_forget()

        showinfo(title="Message",
                 message="Document uploaded successfully!", parent=self)
        return

    def set_threshold(self):
        global score_threshold
        global ranking_threshold
        print(self.score_entry.get())
        if self.score_entry.get():
            if float(self.score_entry.get()) > 100.0 or float(self.score_entry.get()) < 0.0:
                showinfo(title="Warning",
                         message="Invalid Threshold!", parent=self)
                return
            score_threshold = float(self.score_entry.get())/100
        if self.rank_entry.get():
            ranking_threshold = int(self.rank_entry.get())
        self.destroy()

    def open_file_threading(self):
        thread = Thread(target=self.open_file_excel,
                        args=(), daemon=True)
        thread.start()

    def open_file_excel(self):

        filename = askopenfilename(
            title="Open a file", initialdir="/", filetypes=[("Excel files", "*.xlsx")], parent=self)
        self.filename = filename

        if self.filename:
            self.progressbar.pack(padx=20, pady=(10, 0))
            self.progressbar.start()
            uos_outline = pd.read_excel(self.filename)
            uos_html = []
            try:
                for i in range(0, len(uos_outline)):
                    uos_html.append(uos_prefix + "/" + uos_outline["Unit of study code"][i] + "/" + str(
                        uos_outline["Year"][i]) + "-" + uos_outline["Session code"][i] + "-" + uos_outline["Occurrence code"][i])
            except KeyError:
                showinfo(
                    title="Error", message="The excel does not have all the required values [Unit of study code, Year, Session code, Occurrence code]", parent=self)
                return
            loader = AsyncHtmlLoader(uos_html)
            docs = loader.load()
            html2text = Html2TextTransformer()
            docs_transformed = html2text.transform_documents(docs)
            for doc in docs_transformed:
                code = doc.metadata["source"].split("/")[4]
                file_path = os.path.join(directory, code + ".txt")
                # upload the weekly schedule to pinecone, need to change to f.write(result)
                # result = chain.run(
                #     input_documents=[doc], question="Give me a list of topics in the weekly schedule. If does not exist, give me a Short summary of the learning outcomes of this unit of study")
                f = open(file_path, "w", encoding='utf-8')
                f.write(doc.page_content)
                f.close()

            loader = DirectoryLoader(directory)
            documents = loader.load()
            Pinecone.from_documents(
                documents, embeddings_model, index_name=index_name)

            file_paths = glob.glob(os.path.join(directory, "*"))
            for file_path in file_paths:
                os.remove(file_path)

            self.progressbar.pack_forget()

            showinfo(title="Message",
                     message="All documents uploaded successfully!", parent=self)

    def delete_all(self):
        index = Pinecone.from_existing_index(
            "document-similarity", embeddings_model)
        index.delete(delete_all=True)
        showinfo(title="Message",
                 message="Documents deleted successfully!", parent=self)

    def preprocessing(self, text_string):
        result = text_string.translate(
            (str.maketrans('', ''), string.punctuation))
        return " ".join(result.split())


class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.width = 1100
        self.height = 750
        self.title("Document Question Answering")
        self.geometry(f"{1100}x{750}")

        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(1, weight=1)

        self.filename = ""
        self.user_questions = []
        self.document_frames = {}

        if os.path.exists("test") and os.path.getsize("test") > 0:
            with open("test", "r") as file:
                self.user_questions = json.load(file)
        else:
            f = open("test", "w+")
            f.close()

        if not os.path.exists(directory):
            os.makedirs(directory)

        image_path = os.path.join(os.path.dirname(
            os.path.realpath(__file__)), "icons")
        self.history_image = ctk.CTkImage(Image.open(
            os.path.join(image_path, "delete.png")), size=(20, 20))
        self.folder_image = ctk.CTkImage(Image.open(
            os.path.join(image_path, "folder.png")), size=(20, 20))
        self.setting_image = ctk.CTkImage(Image.open(
            os.path.join(image_path, "setting.png")), size=(20, 20))
        self.user_image = ctk.CTkImage(Image.open(
            os.path.join(image_path, "user.png")), size=(25, 25))
        self.bot_image = ctk.CTkImage(Image.open(
            os.path.join(image_path, "bot.png")), size=(28, 28))
        self.history_image = ctk.CTkImage(Image.open(
            os.path.join(image_path, "comment.png")), size=(18, 18))
        self.default_image = ctk.CTkImage(Image.open(os.path.join(
            image_path, "image_icon_light.png")), size=(20, 20))
        self.document_image = ctk.CTkImage(Image.open(os.path.join(
            image_path, "docs.png")), size=(20, 20))

        self.sidebar_frame = ctk.CTkFrame(self)
        self.sidebar_frame.grid(row=0, column=0, rowspan=999, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(4, weight=1)
        self.scrollable_frame = ctk.CTkScrollableFrame(
            self.sidebar_frame, label_text="Loaded Documents")
        self.scrollable_frame.grid(row=0, column=0, rowspan=3, sticky="nsew")
        self.document_button = ctk.CTkButton(self.scrollable_frame, corner_radius=0, height=30, text="name", image=self.history_image,
                                             fg_color="transparent", text_color=("gray10", "gray90"), hover_color=("gray70", "gray30"), anchor="w", command=partial(self.document_button_event, "name"))

        self.scrollable_frame_user_q = ctk.CTkScrollableFrame(
            self.sidebar_frame, label_text="Previous Questions")
        self.scrollable_frame_user_q.grid(
            row=3, column=0, pady=10, sticky="nsew")
        self.scrollable_frame_user_buttons = []
        if len(self.user_questions) > 0:
            for index, item in enumerate(self.user_questions):
                question_button = ctk.CTkButton(self.scrollable_frame_user_q, corner_radius=0, height=30, text=item, image=self.history_image,
                                                fg_color="transparent", text_color=("gray10", "gray90"), hover_color=("gray70", "gray30"), anchor="w", command=partial(self.home_button_event, item))
                question_button.grid(row=index, column=0,
                                     pady=(0, 10), sticky="ew")
                question_button._text_label.configure(
                    wraplength=165, justify="left")
                self.scrollable_frame_user_buttons.append(question_button)

        self.clear_history_button = ctk.CTkButton(
            self.sidebar_frame, text="Clear History", image=self.history_image, anchor="w", command=self.clear_history)
        self.clear_history_button.grid(row=4, column=0)

        self.file_input_button = ctk.CTkButton(
            self.sidebar_frame, text="Compare Unit", image=self.folder_image, anchor="w", command=self.open_file)
        self.file_input_button.grid(row=5, column=0, pady=(10, 10))

        self.setting_button = ctk.CTkButton(
            self.sidebar_frame, text="Settings", image=self.setting_image, anchor="w", command=self.open_toplevel)
        self.setting_button.grid(row=6, column=0, padx=20, pady=(10, 0))
        self.toplevel_window = None

        self.appearance_mode_label = ctk.CTkLabel(
            self.sidebar_frame, text="Appearance Mode:", anchor="w")
        self.appearance_mode_label.grid(row=7, column=0, padx=20, pady=(10, 0))
        self.appearance_mode_optionemenu = ctk.CTkOptionMenu(self.sidebar_frame, values=["Dark", "Light", "System"],
                                                             command=self.change_appearance_mode_event)
        self.appearance_mode_optionemenu.grid(
            row=8, column=0, padx=20, pady=(10, 20))
        self.main_title = ctk.CTkLabel(
            self, text="Document Question Answering", font=ctk.CTkFont(size=20, weight="bold"))
        self.main_title.grid(row=0, column=1, columnspan=2,
                             padx=20, pady=20, sticky="nsew")
        self.scrollable_frame_qa = ctk.CTkScrollableFrame(self, width=400)
        self.scrollable_frame_qa.grid(
            row=1, column=1, columnspan=2, rowspan=2, padx=20, pady=5, sticky="nsew")

        self.progressbar_1 = ctk.CTkProgressBar(self)
        self.progressbar_1.configure(mode="indeterminnate")

        stringvar1 = ctk.StringVar(self)
        stringvar1.trace("w", self.entry_check)
        self.entry = ctk.CTkEntry(
            self, height=38, placeholder_text="Please Enter Your Question Here...", textvariable=stringvar1)
        self.entry.grid(row=4, column=1, padx=(20, 0),
                        pady=(10, 20), sticky="nsew")

        self.main_button_1 = ctk.CTkButton(master=self, fg_color="transparent", border_width=2, text_color=(
            "gray10", "#DCE4EE"), text="Enter", command=self.ask_question, state="disabled")
        self.main_button_1.grid(row=4, column=2, padx=(
            20, 20), pady=(10, 20), sticky="nsew")

    def answer_frame(self, content):
        self.question_frame = ctk.CTkFrame(self.scrollable_frame_qa, fg_color=(
            "gray30", "gray10"), width=1080, corner_radius=10)
        self.bot_image_label = ctk.CTkLabel(
            self.question_frame, text="", image=self.bot_image, corner_radius=5, anchor="w")
        self.bot_image_label.grid(sticky="ne", pady=(4, 0))
        self.message = ctk.CTkLabel(master=self.question_frame, text=content, font=ctk.CTkFont(
            size=15, weight="normal"), anchor="w", justify=ctk.LEFT, wraplength=760, fg_color=("gray30", "gray10"), width=1080)
        self.message.grid(row=0, column=1, rowspan=2, padx=5, pady=5)
        self.question_frame.pack(pady=(0, 5))

    def open_file(self):
        filename = askopenfilename(
            title="Open a file", initialdir="/", filetypes=[("PDF files", "*.pdf")])
        self.filename = filename
        if self.filename:
            # showinfo(title="Message", message="File input successed")
            name = self.filename.split("/")[-1].split(".")[0]
            doc_name = self.filename.split("/")[-1].split(".")[0]
            document_frame_qa = ctk.CTkScrollableFrame(self, width=400)
            self.document_button.configure(fg_color="transparent")
            self.scrollable_frame_qa.grid_forget()
            self.scrollable_frame_qa = document_frame_qa
            self.scrollable_frame_qa.grid(
                row=1, column=1, columnspan=2, rowspan=2, padx=20, pady=5, sticky="nsew")

            if name in self.document_frames.keys():
                doc_name = name + \
                    random.choice(numbers) + random.choice(numbers)

            document_button = ctk.CTkButton(self.scrollable_frame, corner_radius=0, height=30, width=200, text=name, image=self.document_image,
                                            fg_color=("gray75", "gray25"), text_color=("gray10", "gray90"), hover_color=("gray70", "gray30"), anchor="w", command=partial(self.document_button_event, doc_name))
            document_button.grid(row=len(self.document_frames), column=0,
                                 pady=(0, 10), sticky="ew")
            document_button._text_label.configure(
                wraplength=160, justify="left")
            self.document_button = document_button
            self.document_frames[doc_name] = [
                document_frame_qa, document_button, filename]

            self.progressbar_1.grid(row=3, column=1, columnspan=2, padx=(
                20, 20), pady=(5, 0), sticky="ew")
            self.progressbar_1.start()
            thread = Thread(target=self.find_similarity_scores,
                            args=(filename, name, ), daemon=True)
            thread.start()

    def find_similarity_scores(self, filename, name):
        index = Pinecone.from_existing_index(
            "document-similarity", embeddings_model)
        print(filename)

        loader = PyMuPDFLoader(filename)
        data = loader.load()
        if len(data[0].page_content) == 0:
            overview_input = "Invalid PDF file! [Does not contain any text or only contain images]"
            self.progressbar_1.grid_forget()
            self.answer_frame(overview_input)
            return
        text = ""
        for page in data:
            text = text + page.page_content
        text = self.preprocessing(text)
        overview_input = "The input unit (" + name + "):\n\n" + chain.run(
            input_documents=data, question="Short summary of the learning outcomes of this unit of study")
        self.answer_frame(overview_input)

        similar_docs = index.similarity_search_with_score(
            text, k=ranking_threshold)
        if len(similar_docs) == 0:
            self.answer_frame(
                "Sorry there are no unit outlines in the database!")
            self.progressbar_1.grid_forget()
            return
        if similar_docs[0][1] < score_threshold:
            relevance_result = "Sorry, there are no relevant units! Below will show the most relevant units which is below the threshold:"
            self.answer_frame(relevance_result)
        else:
            relevance_result = "Relevant Unit(s):"
            self.answer_frame(relevance_result)
        for i in range(0, ranking_threshold):
            relevance_result = str(i+1) + ".  " + similar_docs[i][0].metadata["source"].split(
                "\\")[-1].split(".")[0] + " --- " + "{:.1%}".format(similar_docs[i][1])
            doc = Document(
                page_content=similar_docs[i][0].page_content[:-8000], metadata=similar_docs[i][0].metadata)
            relevance_result = relevance_result + "\n\n" + \
                chain.run(input_documents=[
                    doc], question="Short summary of the learning outcomes of this unit of study")
            self.answer_frame(relevance_result)

        self.progressbar_1.grid_forget()

    def home_button_event(self, text):
        self.entry.delete(0, ctk.END)
        self.entry.insert(0, text)
        return

    def change_appearance_mode_event(self, new_appearance_mode: str):
        ctk.set_appearance_mode(new_appearance_mode)

    def ask_question(self):
        self.question_frame = ctk.CTkFrame(self.scrollable_frame_qa, fg_color=(
            "gray30", "gray10"), width=1080, corner_radius=10)
        if not self.filename:
            self.bot_image_label = ctk.CTkLabel(
                self.question_frame, text="", image=self.bot_image, corner_radius=5, anchor="w")
            self.bot_image_label.grid(sticky="ne", pady=(4, 0))
            self.message = ctk.CTkLabel(master=self.question_frame, text="Please input your document!", font=ctk.CTkFont(
                size=15, weight="normal"), anchor="w", justify=ctk.LEFT, wraplength=760, fg_color=("gray30", "gray10"), width=1080)
            self.message.grid(row=0, column=1, rowspan=2, padx=5, pady=5)
            self.question_frame.pack(pady=(0, 5))
            return
        self.user_image_label = ctk.CTkLabel(
            self.question_frame, text="", image=self.user_image, corner_radius=5, anchor="s")
        self.user_image_label.grid(row=0, column=0)
        self.question_frame.configure(fg_color=("gray50", "gray25"))
        self.progressbar_1.grid(row=3, column=1, columnspan=2, padx=(
            20, 20), pady=(5, 0), sticky="ew")
        self.progressbar_1.start()
        prompt = self.entry.get()
        question = prompt
        self.message = ctk.CTkLabel(master=self.question_frame, text=question, font=ctk.CTkFont(
            size=15, weight="normal"), anchor="w", justify=ctk.LEFT,  width=1080, wraplength=760)
        self.message.grid(row=0, column=1, rowspan=2, padx=5, pady=5)
        self.question_frame.pack(pady=(0, 5))
        self.entry.delete(0, ctk.END)
        try:
            self.user_questions.insert(0, self.user_questions.pop(
                self.user_questions.index(prompt)))
        except:
            self.user_questions.insert(0, prompt)
        with open("test", "w") as file:
            json.dump(self.user_questions, file)
        for item in self.scrollable_frame_user_buttons:
            item.grid_forget()
        self.scrollable_frame_user_buttons.clear()
        if len(self.user_questions) > 0:
            for index, item in enumerate(self.user_questions):
                question_button = ctk.CTkButton(self.scrollable_frame_user_q, corner_radius=0, height=30, image=self.history_image, border_spacing=10, text=item,
                                                fg_color="transparent", text_color=("gray10", "gray90"), hover_color=("gray70", "gray30"), anchor="w", command=partial(self.home_button_event, item))
                question_button.grid(row=index, column=0, sticky="ew")
                question_button._text_label.configure(
                    wraplength=165, justify="left")
                self.scrollable_frame_user_buttons.append(question_button)
        thread = Thread(target=self.question_result,
                        args=(question, ), daemon=True)
        thread.start()

    def ask_question_event(self, event):
        self.ask_question()

    def answer_question(self, file, question):
        loader = PyMuPDFLoader(file)
        data = loader.load()
        # source document retriever
        # documents = loader.load()
        # text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
        # texts = text_splitter.split_documents(documents)
        # embeddings = OpenAIEmbeddings()
        # db = Chroma.from_documents(texts, embeddings)
        # retriever = db.as_retriever(
        #     search_type="similarity", search_kwargs={"k": 1})
        # qa = RetrievalQA.from_chain_type(llm=OpenAI(
        # ), chain_type="stuff", retriever=retriever, return_source_documents=True)
        # result = qa({"query": question})
        result = chain.run(input_documents=data, question=question)
        return result

    def question_result(self, question):
        if self.filename:
            result = self.answer_question(
                file=self.filename, question=question)
            answer = result
            # answer = result["result"] + "\n\n" + "relevant source text:" + '\n--------------------------------------------------------------------\n'.join(doc.page_content for doc in result["source_documents"])
            print(answer)
            self.answer_frame(answer)
            self.progressbar_1.grid_forget()

    def open_toplevel(self):
        if self.toplevel_window is None or not self.toplevel_window.winfo_exists():
            self.toplevel_window = ToplevelWindow(self)
            self.toplevel_window.attributes('-topmost', 'true')
            self.toplevel_window.geometry(
                "%dx%d+%d+%d" % (500, 400, self.winfo_x() + self.width/4, self.winfo_y() + self.height/4))
        else:
            self.toplevel_window.focus()

    def clear_history(self):
        for item in self.scrollable_frame_user_buttons:
            item.grid_forget()
        self.user_questions.clear()
        open("test", "w").close()
        self.scrollable_frame_user_buttons.clear()

    def entry_check(self, *args):
        if self.entry.get():
            self.main_button_1.configure(state='normal')
            self.entry.unbind('<Return>')
            self.entry.bind('<Return>', self.ask_question_event)
        else:
            self.main_button_1.configure(state='disabled')
            self.entry.unbind('<Return>')

    def document_button_event(self, name):
        self.document_button.configure(fg_color="transparent")
        self.document_button = self.document_frames[name][1]
        self.scrollable_frame_qa.grid_forget()
        self.scrollable_frame_qa = self.document_frames[name][0]
        self.scrollable_frame_qa.grid(
            row=1, column=1, columnspan=2, rowspan=2, padx=20, pady=5, sticky="nsew")
        self.document_button.configure(fg_color=("gray75", "gray25"))
        self.filename = self.document_frames[name][2]

    def preprocessing(self, text_string):
        result = text_string.translate(
            (str.maketrans('', ''), string.punctuation))
        return " ".join(result.split())


if __name__ == "__main__":
    app = App()
    app.resizable(False, False)
    app.mainloop()

    del app
