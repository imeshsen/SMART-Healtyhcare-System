FROM maven:3.8.6-openjdk-8 

RUN useradd -m myuser

USER myuser

WORKDIR /app

COPY . ./

RUN mvn package

COPY serviceAccKey.json /target

CMD ["java" "-jar" "smart-backend-0.0.1-SNAPSHOT.jar"]

# CMD [ "mvn" , "spring-boot:run" ]
