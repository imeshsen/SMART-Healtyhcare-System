import { initializeApp } from "firebase/app";
import { getAuth } from "firebase/auth";
import {getStorage} from "firebase/storage"

const firebaseConfig = {
  apiKey: "AIzaSyC8ZhPCUc_rdB8ujfr9kvL1XV-d3bf08n4",

  authDomain: "smart-healthcare-system-666e1.firebaseapp.com",

  projectId: "smart-healthcare-system-666e1",

  storageBucket: "smart-healthcare-system-666e1.appspot.com",

  messagingSenderId: "895658645483",

  appId: "1:895658645483:web:87e1593fe4edb3b9a8ebae",
};
const app = initializeApp(firebaseConfig);

export const auth = getAuth(app);

export const storage = getStorage(app);
