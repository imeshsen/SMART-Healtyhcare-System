import axios from "axios";
import { signInWithEmailAndPassword } from "firebase/auth";
import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import { auth } from "./Firebase-Config";
import "./Lgin.css";
import image from "../assets/image2.jpg";
import HomeNavbar from "./HomeNavbar";

export default function Lgin() {
  const navigate = useNavigate();

  const login = (e) => {
    signInWithEmailAndPassword(auth, email, password)
      .then((userCredential) => {
        // Signed in
        const user = userCredential.user;
        alert("Success!");
        navigate(`/user/${nic}`);
      })
      .catch((error) => {
        const errorCode = error.code;
        const errorMessage = error.message;
        alert("Please check your username or password!");
      });
  };

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [nic, setNic] = useState("");

  return (
    <div
      className="login"
      style={{
        backgroundImage: `url(${image})`,
        height: "100vh",
        fontSize: "15px",
        backgroundSize: "cover",
        backgroundRepeat: "no-repeat",
      }}
    >
      <HomeNavbar />
      <div className="form">
        <h1>Login</h1>
        <input
          type="NIC"
          placeholder="Enter your NIC"
          value={nic}
          onChange={(e) => setNic(e.target.value)}
          required
        ></input>
        <br />
        <br />
        <input
          type="text"
          placeholder="Email"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
        ></input>
        <br />
        <br />
        <input
          type="password"
          placeholder="Password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
        ></input>
        <br />
        <br />
        <button onClick={login} className="btnlgin">
          Sign in
        </button>
      </div>
    </div>
  );
}
