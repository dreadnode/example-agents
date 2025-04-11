FROM mongo:6.0

ARG FLAG

COPY db.js /docker-entrypoint-initdb.d/db.js

RUN sed -i "s/\*\*FLAG\*\*/$FLAG/g" /docker-entrypoint-initdb.d/db.js