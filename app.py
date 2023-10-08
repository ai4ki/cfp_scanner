import json
import gc

import streamlit as st

from source.source_local import (load_sentence_transformer,
                                 get_mail_data,
                                 preprocess_nl,
                                 encode_calls,
                                 evaluate_calls,
                                 keyword_check,
                                 ask_llm,
                                 format_call,
                                 get_had_table,
                                 format_had_link)

# Set LLM for final evaluation
LLM = "gpt-3.5-turbo"

# Define available newsletter abos
NL_ABOS = ["service.bund.de", "Bundesanzeiger", "HAD"]

# Enable WALL-E
gc.enable()

# Load page text
with open("./assets/page_text.txt", "r", encoding="utf-8") as f:
    page_text = json.load(f)

# Global streamlit settings
st.set_page_config(layout="wide",
                   page_title="Ausschreibungsscanner",
                   page_icon="./images/icon.png",
                   initial_sidebar_state="collapsed",
                   menu_items={"About": page_text["about"]})

with open("./css/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

cols = st.columns([1, 2], gap="large")


def del_data():
    if "mail_data" in st.session_state:
        del st.session_state["mail_data"]
    if "had_table" in st.session_state:
        del st.session_state["had_table"]


# Build the page sidebar
with st.sidebar:
    st.markdown("## Einstellungen")
    st.markdown("#### Wähle eine Quelle:")
    nl_abo = st.selectbox(label="select:",
                          options=NL_ABOS,
                          index=0,
                          on_change=del_data,
                          label_visibility="collapsed")
    source_type = "Newsletter"
    if nl_abo == "HAD":
        source_type = "Ausschreibungen"
    st.markdown(f"#### {source_type} der letzten...")
    day_range = st.selectbox(label="day range",
                             options=["7 Tage", "14 Tage", "21 Tage"],
                             index=1,
                             on_change=del_data,
                             label_visibility="collapsed")
    st.markdown("#### Anzahl an Treffern")
    top_k_default = 5
    if nl_abo == "Bundesanzeiger":
        top_k_default = 3
    top_k = st.slider("top_k", 1, 15, top_k_default, label_visibility="collapsed")
    with st.expander("Was steckt dahinter?"):
        st.markdown(page_text["top-k"])
    st.markdown("#### Bewertung durch Sprachmodell")
    llm_flag = st.radio(label="llm evaluation",
                        options=["nein", "streng", "locker"],
                        index=0,
                        horizontal=True,
                        label_visibility="collapsed")
    with st.expander("Was steckt dahinter?"):
        st.markdown(page_text["llm"])

# Set threshold for LLM evaluation
llm_threshold = 4 if llm_flag == "streng" else 2

# Load data if not already in session state
if nl_abo == "HAD":
    if "had_table" not in st.session_state:
        try:
            st.session_state.had_table = get_had_table(int(day_range.split()[0]))
        except ConnectionError:
            st.warning("Connection to HAD platform failed -- please rerun the app!")
        except:
            st.warning("We have trouble fetching HAD data -- please rerun the app!")

    if not st.session_state.had_table.empty:
        n_calls = len(st.session_state.had_table)

        # Build the page's first column: display HAD table
        with cols[0]:
            st.markdown(f"### Hessische Ausschreibungsdatenbank")
            st.markdown(f"**{n_calls} Ausschreibungen**")
            st.dataframe(st.session_state.had_table)
            st.markdown("")
            had_analyze = st.button("HAD auswerten")

    # Build the page's second column: evaluation results
    if had_analyze:
        # Load titles, queries, and descriptions of research departments
        with open("./assets/departments.json", "r", encoding="utf-8") as f:
            departments = json.load(f)

        # Make a copy of the newsletter dataframe
        filtered_had_table = st.session_state.had_table.copy()

        # Delete calls with negative keywords in the call title
        condition = filtered_had_table.call.apply(lambda x: keyword_check(x))
        filtered_had_table = filtered_had_table.loc[condition]
        n_calls_filtered = len(filtered_had_table.index)

        # load transformer models
        with cols[1]:
            with st.spinner("Wir laden das Modell -- kann bisschen dauern ...:coffee:"):
                try:
                    encoder = load_sentence_transformer()
                except:
                    st.warning("Some error occured while loading the model :(")

        # Encode calls once with Bi-Encoder
        encode_calls(filtered_had_table, encoder)

        # Take care of (rare) cases where number of calls is smaller than top_k
        top_k_be = min(top_k, n_calls_filtered)
        with cols[1]:
            st.markdown("### Auswertung")
            tabs = st.tabs(["FF1", "FF2", "FF3", "FF4", "FF5"])
            with st.spinner(f"...läuft...:rocket:"):
                for i, dept in enumerate(departments):
                    results_df = evaluate_calls(filtered_had_table, top_k_be, dept["query"], encoder)
                    eval_llm = ""
                    one_call_passed = False

                    with tabs[i]:
                        st.markdown(f'##### {dept["name"]}')
                        st.markdown(f'*{dept["query"]}*')
                        for idx, rows in results_df.sort_values('be_scores', ascending=False).iterrows():
                            client_data = f"**HAD-Ausschreibung {idx}:**<br>{rows.call[:400]}...<br>*{rows.client_info}*"
                            call_link = format_had_link(rows.link)

                            if llm_flag == "nein":
                                st.markdown(f"{client_data} {call_link}<br>", unsafe_allow_html=True)
                            else:
                                eval_llm = ask_llm(LLM, dept["description"], rows.call)
                                if eval_llm.isnumeric():
                                    if int(eval_llm) >= llm_threshold:
                                        st.markdown(f"{client_data} {call_link}<br>", unsafe_allow_html=True)
                                        one_call_passed = True
                                else:
                                    st.markdown(
                                        f"{client_data}**Achtung: Es erfolgte keine Bewertung durch das Sprachmodell!**",
                                        unsafe_allow_html=True)

                        if llm_flag != "nein" and not one_call_passed:
                            st.markdown("Im gewählten Zeitraum gab es für das Forschungsfeld keine passenden Ausschreibungen.")

                    del results_df

            del (
                encoder,
                filtered_had_table,
                condition,
                tabs,
                page_text,
                departments
            )

            gc.collect()

else:
    if "mail_data" not in st.session_state:
        try:
            st.session_state.mail_data = get_mail_data(nl_abo, int(day_range.split()[0]))
        except ConnectionError:
            st.warning("Connection to email server failed -- please rerun the app!")
        except:
            st.warning("We have trouble downloading your emails -- please rerun the app!")

    if st.session_state.mail_data:
        n_newsletters = st.session_state.mail_data["total"]
        newsletters = [st.session_state.mail_data["data"][nl]["date"] for nl in range(n_newsletters)]

        # Build the page's first column: select a newsletter and display as table
        with cols[0]:
            st.markdown(f"### {nl_abo}")
            st.markdown("###### Wähle einen Newsletter:")
            nl_selected = st.selectbox("Select Newsletter", reversed(newsletters), label_visibility="collapsed")
            nl_index = newsletters.index(nl_selected)

            nl_content = st.session_state.mail_data["data"][nl_index]["content"]
            nl_as_df, nl_call_list = preprocess_nl(nl_content, nl_abo)
            n_calls = len(nl_as_df.index)

            st.markdown(f"###### {n_calls} Ausschreibungen:")
            st.dataframe(nl_as_df)

            st.markdown("")
            nl_analyze = st.button("Newsletter auswerten")

    # Build the page's second column: evaluation results
    if nl_analyze:
        # Load titles, queries, and descriptions of research departments
        with open("./assets/departments.json", "r", encoding="utf-8") as f:
            departments = json.load(f)

        # Make a copy of the newsletter dataframe
        filtered_df = nl_as_df.copy()

        # Delete calls with negative keywords in the call title
        condition = filtered_df.call.apply(lambda x: keyword_check(x))
        filtered_df = filtered_df.loc[condition]
        n_calls_filtered = len(filtered_df.index)

        # load transformer models
        with cols[1]:
            with st.spinner("Wir laden das Modell -- kann bisschen dauern ...:coffee:"):
                try:
                    encoder = load_sentence_transformer()
                except:
                    st.warning("Some error occured while loading the model :(")

        # Encode calls once with Bi-Encoder
        encode_calls(filtered_df, encoder)

        # Take care of (rare) cases where number of calls is smaller than top_k
        top_k_be = min(top_k, n_calls_filtered)

        with cols[1]:
            st.markdown("### Auswertung")
            tabs = st.tabs(["FF1", "FF2", "FF3", "FF4", "FF5"])
            with st.spinner("... läuft...:rocket:"):
                for i, dept in enumerate(departments):
                    results_df = evaluate_calls(filtered_df, top_k_be, dept["query"], encoder)
                    eval_llm = ""
                    one_call_passed = False

                    with tabs[i]:
                        st.markdown(f'##### {dept["name"]}')
                        st.markdown(f'*{dept["query"]}*')
                        for idx, rows in results_df.sort_values('be_scores', ascending=False).iterrows():
                            if nl_abo == "service.bund.de":
                                client_data = format_call(nl_call_list[idx])
                            elif nl_abo == "Bundesanzeiger":
                                client_data = nl_call_list[idx]

                            if llm_flag == "nein":
                                st.markdown(f"{client_data}", unsafe_allow_html=True)
                            else:
                                eval_llm = ask_llm(LLM, dept["description"], rows.call)
                                if eval_llm.isnumeric():
                                    if int(eval_llm) >= llm_threshold:
                                        st.markdown(f"{client_data}", unsafe_allow_html=True)
                                        one_call_passed = True
                                else:
                                    st.markdown(
                                        f"{client_data}**Achtung: Es erfolgte keine Bewertung durch das Sprachmodell!**",
                                        unsafe_allow_html=True)

                        if llm_flag != "nein" and not one_call_passed:
                            st.markdown("In dieser Ausgabe gab es für das Forschungsfeld keine passenden Ausschreibungen.")

                    del results_df

            del (
                encoder,
                filtered_df,
                nl_as_df,
                condition,
                tabs,
                nl_content,
                nl_call_list,
                page_text,
                departments
            )

            gc.collect()