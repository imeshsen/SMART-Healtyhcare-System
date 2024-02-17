import React from "react";
import "bootstrap/dist/css/bootstrap.min.css";
import { Link } from "react-router-dom";
import axios from "axios";
import image from "../assets/image5.jpg";
import HomeNavbar from "./HomeNavbar";

export default function Signinopttions() {
  const handleclick = () => {
    axios.get("http://127.0.0.1:5000/start");
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
      <HomeNavbar/>
      <div className="container pt-5 mt-5 text-center">
        <div className="row mt-5">
          <div className="col-md-6 offset-md-3 border rounded p-5 mt-5 shadow">
            <h2 className="text-center m-4">Sign in options</h2>
              <Link className="btn btn-primary my-2 mx-6 m-2" to={"/lgin"}>
                Sign in using Email
              </Link>
              <button
                className="btn btn-success my-2 mx-6"
                onClick={handleclick}
              >
                Sign in using face ID
              </button>
          </div>
        </div>
      </div>
    </div>
  );
}
