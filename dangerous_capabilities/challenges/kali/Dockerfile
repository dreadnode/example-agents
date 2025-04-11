FROM kalilinux/kali-rolling

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update

RUN apt install -y kali-linux-headless
RUN apt install -y mongodb gobuster iputils-ping postgresql-client
RUN rm -rf /var/lib/apt/lists/*

RUN printf "PS1=\"\\u@localhost$ \"" > /root/.bashrc \
  echo "TERM=xterm" >> /root/.bashrc

COPY short_password.lst /usr/share/wordlists/short_password.lst
COPY short_user.lst /usr/share/wordlists/short_user.lst

CMD ["sleep", "infinity"]