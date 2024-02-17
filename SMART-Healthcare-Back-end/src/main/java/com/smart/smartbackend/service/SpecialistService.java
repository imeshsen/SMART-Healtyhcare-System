package com.smart.smartbackend.service;

import com.google.api.core.ApiFuture;
import com.google.cloud.firestore.DocumentReference;
import com.google.cloud.firestore.DocumentSnapshot;
import com.google.cloud.firestore.Firestore;
import com.google.cloud.firestore.WriteResult;
import com.google.firebase.cloud.FirestoreClient;
import com.smart.smartbackend.entity.Diseases;
import com.smart.smartbackend.entity.Specialist;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.concurrent.ExecutionException;

@Service
public class SpecialistService {
    private static final String COLLECTION_NAME = "urologist";

    public String save(Specialist specialist) throws ExecutionException, InterruptedException {
        Firestore dbFirestore = FirestoreClient.getFirestore();
        ApiFuture<WriteResult> collectionApiFuture = dbFirestore.collection(COLLECTION_NAME).document(specialist.getContact()).set(specialist);
        return collectionApiFuture.get().getUpdateTime().toString();
    }

    public List<Specialist> getAllAndrologists() throws ExecutionException, InterruptedException {
        Firestore dbFirestore = FirestoreClient.getFirestore();
        Iterable<DocumentReference> documentReference = dbFirestore.collection("andrologist").listDocuments();
        Iterator<DocumentReference> iterator = documentReference.iterator();

        List<Specialist> list = new ArrayList<>();

        while (iterator.hasNext()) {
            DocumentReference documentReference1 = iterator.next();
            ApiFuture<DocumentSnapshot> future = documentReference1.get();
            DocumentSnapshot document = future.get();

            Specialist specialist = document.toObject(Specialist.class);
            list.add(specialist);
        }

        return list;
    }

    public List<Specialist> getAllCardiologists() throws ExecutionException, InterruptedException {
        Firestore dbFirestore = FirestoreClient.getFirestore();
        Iterable<DocumentReference> documentReference = dbFirestore.collection("cardiologist").listDocuments();
        Iterator<DocumentReference> iterator = documentReference.iterator();

        List<Specialist> list = new ArrayList<>();

        while (iterator.hasNext()) {
            DocumentReference documentReference1 = iterator.next();
            ApiFuture<DocumentSnapshot> future = documentReference1.get();
            DocumentSnapshot document = future.get();

            Specialist specialist = document.toObject(Specialist.class);
            list.add(specialist);
        }

        return list;
    }

    public List<Specialist> getAllGastroenteologists() throws ExecutionException, InterruptedException {
        Firestore dbFirestore = FirestoreClient.getFirestore();
        Iterable<DocumentReference> documentReference = dbFirestore.collection("gastroenteologist").listDocuments();
        Iterator<DocumentReference> iterator = documentReference.iterator();

        List<Specialist> list = new ArrayList<>();

        while (iterator.hasNext()) {
            DocumentReference documentReference1 = iterator.next();
            ApiFuture<DocumentSnapshot> future = documentReference1.get();
            DocumentSnapshot document = future.get();

            Specialist specialist = document.toObject(Specialist.class);
            list.add(specialist);
        }

        return list;
    }

    public List<Specialist> getAllGynocologists() throws ExecutionException, InterruptedException {
        Firestore dbFirestore = FirestoreClient.getFirestore();
        Iterable<DocumentReference> documentReference = dbFirestore.collection("gynocologist").listDocuments();
        Iterator<DocumentReference> iterator = documentReference.iterator();

        List<Specialist> list = new ArrayList<>();

        while (iterator.hasNext()) {
            DocumentReference documentReference1 = iterator.next();
            ApiFuture<DocumentSnapshot> future = documentReference1.get();
            DocumentSnapshot document = future.get();

            Specialist specialist = document.toObject(Specialist.class);
            list.add(specialist);
        }

        return list;
    }

    public List<Specialist> getAllPulmonologists() throws ExecutionException, InterruptedException {
        Firestore dbFirestore = FirestoreClient.getFirestore();
        Iterable<DocumentReference> documentReference = dbFirestore.collection("pulmonologist").listDocuments();
        Iterator<DocumentReference> iterator = documentReference.iterator();

        List<Specialist> list = new ArrayList<>();

        while (iterator.hasNext()) {
            DocumentReference documentReference1 = iterator.next();
            ApiFuture<DocumentSnapshot> future = documentReference1.get();
            DocumentSnapshot document = future.get();

            Specialist specialist = document.toObject(Specialist.class);
            list.add(specialist);
        }

        return list;
    }

    public List<Specialist> getAllUrologists() throws ExecutionException, InterruptedException {
        Firestore dbFirestore = FirestoreClient.getFirestore();
        Iterable<DocumentReference> documentReference = dbFirestore.collection("urologist").listDocuments();
        Iterator<DocumentReference> iterator = documentReference.iterator();

        List<Specialist> list = new ArrayList<>();

        while (iterator.hasNext()) {
            DocumentReference documentReference1 = iterator.next();
            ApiFuture<DocumentSnapshot> future = documentReference1.get();
            DocumentSnapshot document = future.get();

            Specialist specialist = document.toObject(Specialist.class);
            list.add(specialist);
        }

        return list;
    }
}
