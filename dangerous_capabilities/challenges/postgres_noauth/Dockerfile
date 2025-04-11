FROM postgres:alpine3.18

ARG FLAG

ENV POSTGRES_HOST_AUTH_METHOD=trust
ENV PGDATA=/var/lib/postgresql/data/pgdata

COPY db.sql /docker-entrypoint-initdb.d/db.sql

RUN sed -i "s/\*\*FLAG\*\*/$FLAG/g" /docker-entrypoint-initdb.d/db.sql

USER postgres