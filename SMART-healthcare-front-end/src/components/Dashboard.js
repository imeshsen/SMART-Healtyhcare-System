import axios from "axios";
import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import "./Dash.css";
import Navbar from "./Navbar";
import image from "../assets/image6.jpg";

export default function () {
  const [name, setName] = useState("");
  const [nic, setNic] = useState("");
  const [mobile, setMobile] = useState("");
  const [hometown, setHometown] = useState("");
  const [error, setError] = useState(false);

  const navigate = useNavigate();

  //const patient = { name, nic, mobile, hometown };

  const handleclick = (e) => {
    e.preventDefault();
    if (
      name.length == 0 ||
      nic.length == 0 ||
      mobile.length == 0 ||
      hometown.length == 0
    ) {
      setError(true);
    }

    if (name && nic && mobile && hometown) {
      const patient = { name, nic, mobile, hometown };
      fetch("http://localhost:8080/api/v1/patient/save", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(patient),
      }).then(() => {
        axios.post("http://127.0.0.1:5000/add", patient);
        // navigate(`/user/${patient.nic}`);
      });
    }
  };

  const facedt = { nic };
  const face = (e) => {
    e.preventDefault();
    fetch("http://127.0.0.1:5000/add", {
      method: "POST",
      headers: { "Content-Type": "text/plain" },
      body: facedt,
    });
  };
  return (
    <div
      style={{
        backgroundImage: `url(${image})`,
        height: "100vh",
        fontSize: "15px",
        backgroundSize: "cover",
        backgroundRepeat: "no-repeat",
      }}
    >
      <Navbar />
      <div className="content">
        <div>
          <h1>Welcome</h1>
          <h3>Fill up your personal information</h3>
          <div>
            <input
              type="text"
              placeholder="Name with initials"
              value={name}
              name="name"
              onChange={(e) => setName(e.target.value)}
            ></input>
            {error && name.length <= 0 ? (
              <label> Email can't be empty</label>
            ) : (
              ""
            )}
            <br />
            <br />
            <input
              type="text"
              placeholder="Nic number"
              value={nic}
              name="nic"
              onChange={(e) => setNic(e.target.value)}
            ></input>
            {error && nic.length <= 0 ? <label> NIC can't be empty</label> : ""}
            <br />
            <br />
            <input
              type="text"
              placeholder="Mobile number"
              value={mobile}
              onChange={(e) => setMobile(e.target.value)}
            ></input>
            {error && mobile.length <= 0 ? (
              <label> Mobile number can't be empty</label>
            ) : (
              ""
            )}
            <br />
            <br />
            <input
              type="text"
              placeholder="Hometown"
              value={hometown}
              onChange={(e) => setHometown(e.target.value)}
            ></input>
            {error && hometown.length <= 0 ? (
              <label> Hometown can't be empty</label>
            ) : (
              ""
            )}
            <br />
            <br />
            <button className="btnsubmit" onClick={handleclick}>
              Submit
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
