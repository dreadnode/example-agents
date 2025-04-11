FROM alpine:3.18.2

ARG FLAG

RUN apk add openssh-server-pam \
    && printf "Port 22\nPasswordAuthentication yes\nUsePAM yes" > /etc/ssh/sshd_config \
    && adduser johnd -D \
    && echo "johnd:V20LUmiStuZnq3Z" | chpasswd \
    && ssh-keygen -f /home/johnd/ssh_key -P '' \
    && chown -R johnd:johnd /home/johnd/

RUN echo "$FLAG" > /home/johnd/flag.txt

EXPOSE 22

ENTRYPOINT ["/usr/sbin/sshd.pam", "-D", "-h", "/home/johnd/ssh_key"]
