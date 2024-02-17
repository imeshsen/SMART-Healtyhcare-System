import React, { useState } from "react";
import { createUserWithEmailAndPassword } from "firebase/auth";
import { auth } from "./Firebase-Config";
import { useNavigate } from "react-router-dom";
import "./Reg.css";
import image from "../assets/home.jpg";
import HomeNavbar from "./HomeNavbar";

export default function Reg() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");

  const navigate = useNavigate();

  const register = (e) => {
    e.preventDefault();
    createUserWithEmailAndPassword(auth, email, password)
      .then((userCredential) => {
        // Signed in
        const user = userCredential.user;
        const patient = { email, password };

        alert("Success!");
        navigate("/dashboard");
      })
      .catch((error) => {
        const errorCode = error.code;
        const errorMessage = error.message;
        alert("Invalid email and password");
        // ..
      });
  };

  return (
    <div className="register1">
      <div
      style={{
        backgroundImage: `url(${image})`,
        height: "100vh",
        fontSize: "15px",
        backgroundSize: "cover",
        backgroundRepeat: "no-repeat",
      }}
    >
      <HomeNavbar />
      <div className="fill">
      <h1>Sign up today</h1>
      <form>
        <input
          type="text"
          value={email}
          name="email"
          placeholder="E mail"
          onChange={(e) => setEmail(e.target.value)}
          required
        ></input>
        <br />
        <br />
        <input
          type="password"
          value={password}
          name="password"
          placeholder="Password"
          onChange={(e) => setPassword(e.target.value)}
          required
        ></input>
        <br />
        <br />
        <button type="submit" className="btn" onClick={register}>
          Sign up
        </button>
      </form>
      </div>
    </div>
    </div>
    
  );
}
