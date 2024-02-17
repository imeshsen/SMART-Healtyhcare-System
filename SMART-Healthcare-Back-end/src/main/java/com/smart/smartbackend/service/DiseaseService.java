package com.smart.smartbackend.service;

import com.google.api.core.ApiFuture;
import com.google.cloud.firestore.*;
import com.google.firebase.cloud.FirestoreClient;
import com.smart.smartbackend.entity.Diseases;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.concurrent.ExecutionException;

@Service
public class DiseaseService {


    private static final String COLLECTION_NAME = "diseases";
    private Object RetrieveDataSnippets;

    public String saveDisease(Diseases diseases) throws ExecutionException, InterruptedException {
        Firestore dbFirestore = FirestoreClient.getFirestore();
        ApiFuture<WriteResult> collectionApiFuture = dbFirestore.collection(COLLECTION_NAME).document(diseases.getName()).set(diseases);
        return collectionApiFuture.get().getUpdateTime().toString();
    }

    public Diseases searchDiseaseName(String name) throws ExecutionException, InterruptedException {
        Firestore dbFirestore = FirestoreClient.getFirestore();
        DocumentReference documentReference = dbFirestore.collection(COLLECTION_NAME).document(name);
        ApiFuture<DocumentSnapshot> future = documentReference.get();
        DocumentSnapshot document = future.get();

        Diseases diseases = null;
        if (document.exists()) {
            diseases = document.toObject(Diseases.class);
            return diseases;
        } else {
            return null;
        }
    }

    public List<Diseases> findAll() throws ExecutionException, InterruptedException {
        Firestore dbFirestore = FirestoreClient.getFirestore();
        Iterable<DocumentReference> documentReference = dbFirestore.collection(COLLECTION_NAME).listDocuments();
        Iterator<DocumentReference> iterator = documentReference.iterator();

        List<Diseases> diseasesList = new ArrayList<>();

        while (iterator.hasNext()) {
            DocumentReference documentReference1 = iterator.next();
            ApiFuture<DocumentSnapshot> future = documentReference1.get();
            DocumentSnapshot document = future.get();

            Diseases diseases = document.toObject(Diseases.class);
            diseasesList.add(diseases);
        }

        return diseasesList;
    }
}
