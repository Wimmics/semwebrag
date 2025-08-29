import subprocess
import sys
import os
import logging

def main():

    logging.basicConfig(
        # filename="project.log",
        # filemode="w",
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
        logging.FileHandler("project.log", mode="w"),
        logging.StreamHandler(sys.stdout)  
    ]
    )

    logging.info("starting server...")

    http_server = subprocess.Popen(
        [sys.executable, "-m", "http.server"],
        cwd=os.getcwd(),
        # stdout=subprocess.PIPE,
        # stderr=subprocess.STDOUT,
        stdout=sys.stdout,
        stderr=sys.stderr,
        text=True
    )

    flask_server = subprocess.Popen(
        [sys.executable, "appflask.py"],
        cwd=os.getcwd(),
        # stdout=subprocess.PIPE,
        # stderr=subprocess.STDOUT,
        stdout=sys.stdout,
        stderr=sys.stderr,
        text=True
    )

    # try:
    #     while True:
    #         http_line = http_server.stdout.readline()
    #         flask_line = flask_server.stdout.readline()

    #         if http_line:
    #             logging.info("[http.server] %s", http_line.strip())
    #         if flask_line:
    #             logging.info("[Flask] %s", flask_line.strip())

    #         if http_server.poll() is not None and flask_server.poll() is not None:
    #             break

    # except KeyboardInterrupt:
    #     logging.info("Interruption ...")
    #     http_server.terminate()
    #     flask_server.terminate()

    # logging.info("Servers stopped.")

if __name__ == "__main__":
    main()
